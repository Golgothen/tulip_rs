use crate::common::{validate_inputs, validate_options};
#[cfg(feature = "simd_options")]
pub use crate::indicator_types::IndicatorByOptions;
#[cfg(any(feature = "simd_assets", feature = "simd_options"))]
pub use crate::indicator_types::SimdIndicatorResult;
pub use crate::indicator_types::{Indicator, IndicatorResult, TIndicatorState, TState};

use crate::indicators::{max::State as MaxState, min::State as MinState};

use crate::types::{Cold, DisplayGroup, DisplayType, IndicatorError, IndicatorType, Info, Warm};
use serde::{Deserialize, Serialize};

/// Number of input price series required by this indicator.
pub const INPUTS: usize = 3;

/// Number of option parameters required by this indicator.
pub const OPTIONS: usize = 1;

/// Streaming state for the Breakout indicator.
///
/// Holds the last `period` highs and lows: the channel for the next bar is measured over
/// the `period` bars *before* it, so one more bar of history is kept than Donchian keeps.
/// Closes are not kept — only the current bar's close is ever compared.
#[derive(Serialize, Deserialize, Clone)]
pub struct IndicatorState {
    high: Vec<f64>,
    low: Vec<f64>,
    state: State<Warm>,
    periods: (usize, usize),
}
impl IndicatorState {
    pub fn new(state: State<Warm>, high: &[f64], low: &[f64], periods: (usize, usize)) -> Self {
        Self {
            high: high[high.len() - periods.0..].to_vec(),
            low: low[low.len() - periods.0..].to_vec(),
            state,
            periods,
        }
    }
}
impl TIndicatorState<INPUTS> for IndicatorState {
    fn batch_indicator(
        &mut self,
        inputs: &[&[f64]; INPUTS],
        _optional_outputs: Option<&[bool]>,
    ) -> Result<Vec<Vec<f64>>, IndicatorError> {
        validate_inputs(inputs, 1)?;

        self.high.extend_from_slice(inputs[0]);
        self.low.extend_from_slice(inputs[1]);
        let (mut lower_line, mut upper_line, mut signal_line) = {
            let capacity = inputs[0].len();
            (
                crate::uninit_vec!(f64, capacity),
                crate::uninit_vec!(f64, capacity),
                crate::uninit_vec!(f64, capacity),
            )
        };
        // The retained history is exactly `period` bars, so the first new bar sits at
        // index `period` of the high/low buffers and at index 0 of the new closes.
        cycle(
            (&self.high, &self.low, inputs[2]),
            self.periods,
            (&mut lower_line, &mut upper_line, &mut signal_line),
            &mut self.state,
        );

        self.high.drain(..self.high.len() - self.periods.0);
        self.low.drain(..self.low.len() - self.periods.0);

        Ok(vec![lower_line, upper_line, signal_line])
    }
}

/// Per-bar state: the rolling lowest low and highest high, as used by Donchian Channel.
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound = "")]
pub struct State<S = Cold> {
    pub min_state: MinState<S>,
    pub max_state: MaxState<S>,
}

impl State {
    /// Seeds the rolling min/max over the first `look_back` bars, exactly as Donchian Channel
    /// does. The first `calc` then evaluates the channel ending at bar `look_back`.
    pub fn init_state(high: &[f64], low: &[f64], look_back: usize) -> State<Warm> {
        let min_state = MinState::init_state(low, look_back);
        let max_state = MaxState::init_state(high, look_back);
        State {
            min_state,
            max_state,
        }
    }
}

/// Classifies a close against the channel of the preceding bars.
///
/// `+1.0` when the close is strictly above the prior highest high, `-1.0` when strictly below
/// the prior lowest low, `0.0` otherwise (including a close equal to either level, and a NaN close).
#[inline(always)]
pub fn calc_signal(close: f64, lower: f64, upper: f64) -> f64 {
    if close > upper {
        1.0
    } else if close < lower {
        -1.0
    } else {
        0.0
    }
}

impl TState for State<Warm> {
    /// `(high, low, close, i, (period, look_back))` — `i` is the index of the bar being
    /// classified in `high`/`low`, `close` is that bar's close. The channel is the one
    /// ending at `i - 1`, so `i` must be at least `period`.
    type Inputs<'a> = (&'a [f64], &'a [f64], f64, usize, (usize, usize));
    /// `(lower, upper, signal)`.
    type Outputs = (f64, f64, f64);

    #[inline(always)]
    fn calc<'a>(
        &mut self,
        (high, low, close, i, periods): Self::Inputs<'a>,
    ) -> (f64, f64, f64) {
        let (lower, _) = self.min_state.calc((low, i - 1, periods));
        let (upper, _) = self.max_state.calc((high, i - 1, periods));

        (lower, upper, calc_signal(close, lower, upper))
    }
    /// Unchecked version of [`calc`] using the min/max helpers' unchecked, chunked search.
    ///
    /// # Safety
    ///
    /// Callers must ensure `period <= i < high.len()` and `low.len() == high.len()`.
    #[inline(always)]
    unsafe fn calc_unchecked(&mut self, inputs: Self::Inputs<'_>) -> Self::Outputs {
        self.calc_chuncked_unchecked::<4>(inputs)
    }
}
impl State<Warm> {
    /// [`calc`] with SIMD-hint size `N` passed to the min/max window searches.
    ///
    /// # Safety
    ///
    /// Callers must ensure `period <= i < high.len()` and `low.len() == high.len()`.
    #[inline(always)]
    pub unsafe fn calc_chuncked_unchecked<const N: usize>(
        &mut self,
        (high, low, close, i, periods): (&[f64], &[f64], f64, usize, (usize, usize)),
    ) -> (f64, f64, f64) {
        let (lower, _) = self
            .min_state
            .calc_chuncked_unchecked::<N>((low, i - 1, periods));
        let (upper, _) = self
            .max_state
            .calc_chuncked_unchecked::<N>((high, i - 1, periods));

        (lower, upper, calc_signal(close, lower, upper))
    }
}
/// Performs the main calculation loop for the Breakout indicator.
///
/// # Arguments
///
/// * `inputs` - `(high, low, close)`. `high`/`low` start `period` bars before the first bar
///   to classify; `close` starts AT the first bar to classify, so `close[j]` pairs with
///   `high[period + j]`.
/// * `periods` - A tuple of `(period, look_back)` where `look_back = period - 1`.
/// * `output_lines` - Mutable slices for `(lower, upper, signal)`.
/// * `state` - The rolling min/max state, positioned to evaluate bar `period - 1` next.
fn cycle(
    (high, low, close): (&[f64], &[f64], &[f64]),
    periods: (usize, usize),
    output_lines: (&mut [f64], &mut [f64], &mut [f64]),
    state: &mut State<Warm>,
) {
    let (lower_line, upper_line, signal_line) = output_lines;

    for (j, i) in (periods.0..high.len()).enumerate() {
        unsafe {
            let c = *close.get_unchecked(j);
            let (lower, upper, signal) = state.calc_unchecked((high, low, c, i, periods));
            *lower_line.get_unchecked_mut(j) = lower;
            *upper_line.get_unchecked_mut(j) = upper;
            *signal_line.get_unchecked_mut(j) = signal;
        }
    }
}

/// Breakout: a close beyond the Donchian channel of the preceding `period` bars.
///
/// At bar `i`, `upper` is the highest high and `lower` the lowest low of bars
/// `i - period ..= i - 1` — the current bar is excluded, which is what lets a close break
/// out of it (a Donchian channel that includes the bar can never be closed above).
/// `signal` is `+1` when `close[i] > upper`, `-1` when `close[i] < lower`, else `0`.
///
/// Needs `period + 1` bars; the first output is for bar `period`.
pub struct Breakout;

impl Indicator<INPUTS, OPTIONS> for Breakout {
    type IndicatorState = IndicatorState;
    const INFO: Info = Info {
        name: "breakout",
        full_name: "Breakout",
        indicator_type: IndicatorType::Trend,
        inputs: &["high", "low", "close"],
        options: &["period"],
        outputs: &["lower", "upper", "signal"],
        optional_outputs: &[],
        display_groups: &[
            DisplayGroup {
                offset: None,
                id: "breakout_channel",
                label: "Breakout Channel",
                display_type: DisplayType::Overlay,
                outputs: &["lower", "upper"],
            },
            DisplayGroup {
                offset: None,
                id: "breakout_signal",
                label: "Breakout Signal",
                display_type: DisplayType::Indicator,
                outputs: &["signal"],
            },
        ],
    };

    fn min_data(options: &[f64; OPTIONS]) -> usize {
        options[0] as usize + 1
    }

    fn indicator(
        inputs: &[&[f64]; INPUTS],
        options: &[f64; OPTIONS],
        _optional_outputs: Option<&[bool]>,
    ) -> IndicatorResult<Self::IndicatorState> {
        validate_options(options)?;

        validate_inputs(inputs, Self::min_data(options))?;

        let periods = (options[0] as usize, options[0] as usize - 1);
        let [high, low, close] = inputs;

        let (mut lower_line, mut upper_line, mut signal_line) = {
            let capacity = Self::output_length(high.len(), options);
            (
                crate::uninit_vec!(f64, capacity),
                crate::uninit_vec!(f64, capacity),
                crate::uninit_vec!(f64, capacity),
            )
        };

        let mut state = State::init_state(high, low, periods.1);
        cycle(
            (high, low, &close[periods.0..]),
            periods,
            (&mut lower_line, &mut upper_line, &mut signal_line),
            &mut state,
        );

        Ok((
            vec![lower_line, upper_line, signal_line],
            IndicatorState::new(state, high, low, periods),
        ))
    }

    #[cfg(feature = "simd_assets")]
    fn indicator_by_assets<const N: usize>(
        inputs: &[&[&[f64]; INPUTS]; N], //stock[ fields [ field [f64] ] ]
        options: &[f64; OPTIONS],
        optional_outputs: Option<&[bool]>,
    ) -> SimdIndicatorResult<Vec<Self::IndicatorState>> {
        crate::indicators::simd_indicators::breakout_simd::indicator_by_assets::<N>(
            inputs,
            options,
            optional_outputs,
        )
    }
}

#[cfg(feature = "simd_options")]
impl IndicatorByOptions<INPUTS, OPTIONS> for Breakout {
    fn indicator_by_options<const N: usize>(
        inputs: &[&[f64]; INPUTS],
        options: &[&[f64; OPTIONS]; N],
        optional_outputs: Option<&[bool]>,
    ) -> SimdIndicatorResult<Vec<Self::IndicatorState>> {
        crate::indicators::simd_indicators::breakout_simd::indicator_by_options::<N>(
            inputs,
            options,
            optional_outputs,
        )
    }
}
