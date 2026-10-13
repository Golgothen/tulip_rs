use crate::common::validate_options;
use crate::common_simd::assets::validate_inputs;
use crate::indicators::breakout::{Breakout, Indicator, IndicatorState, State, INPUTS, OPTIONS};
use crate::indicators::simd_indicators::breakout_simd::{assets::SimdState, TSimdState, TState};
use crate::indicators::simd_indicators::road_train::{Asset, Driver, PrimeMover};
use crate::types::{IndicatorError, Warm};
use std::simd::Simd;

/// SIMD driver that advances the Breakout indicator across `N` asset lanes per scheduling
/// epoch.
struct BreakoutDriver {
    /// `period`: bars of history ahead of the first bar classified in each epoch.
    period: usize,
    /// `period - 1`: the min/max window look-back.
    look_back: usize,
}

impl Driver<State<Warm>> for BreakoutDriver {
    /// Processes one epoch of bars for `N` assets simultaneously using SIMD.
    ///
    /// Reads from `inputs[asset][field]` (high, low, close), writes to
    /// `outputs[asset][output]` (lower, upper, signal), and updates `states[asset]` in place.
    fn next_run<const N: usize>(
        &mut self,
        inputs: Vec<Vec<&[f64]>>,
        mut outputs: Vec<Vec<&mut [f64]>>,
        mut states: Vec<&mut State<Warm>>,
        _options: Vec<Option<&()>>,
    ) {
        let data_len = inputs[0][0].len();

        let outputs = crate::extract_output_ptrs!(outputs, N, lower_ptr, upper_ptr, signal_ptr);
        let inputs = crate::extract_input_ptrs!(inputs, N, high_ptrs, low_ptrs, close_ptrs);
        let mut state = SimdState::from_states(&mut states);

        self.cycle::<N>(inputs, outputs, data_len, &mut state);

        state.write_states(&mut states);
    }
}
impl BreakoutDriver {
    /// Inner SIMD loop for one epoch of the Breakout computation.
    ///
    /// Each epoch's slices start `period` bars before the first bar to classify, so bars
    /// `period..data_len` are classified, each against the channel ending one bar earlier.
    fn cycle<const N: usize>(
        &self,
        inputs: ([*const f64; N], [*const f64; N], [*const f64; N]),
        outputs: ([*mut f64; N], [*mut f64; N], [*mut f64; N]),
        data_len: usize,
        state: &mut SimdState<N>,
    ) {
        let (high_ptrs, low_ptrs, close_ptrs) = inputs;
        let (lower_line_ptr, upper_line_ptr, signal_line_ptr) = outputs;

        for (j, i) in (self.period..data_len).enumerate() {
            let close: Simd<f64, N> = crate::extract_simd_inputs_at_index!(i, N, c @ close_ptrs);
            let (lower, upper, signal) =
                state.calc((high_ptrs, low_ptrs, close, i, self.look_back));

            crate::write_simd_at_indices!(N, j,
                lower_line_ptr => lower,
                upper_line_ptr => upper,
                signal_line_ptr => signal
            );
        }
    }
}

/// Calculates the Breakout indicator for `N` assets simultaneously using SIMD parallelism.
///
/// All assets share the same `options`. Uses the [`PrimeMover`] scheduler to batch assets
/// into SIMD-width groups.
///
/// # Arguments
/// * `inputs` - An array of `N` asset input sets; `inputs[i]` is `[high, low, close]`.
/// * `options` - Shared options applied to all `N` assets: `[period]`.
/// * `_optional_outputs` - Unused; pass `None`.
///
/// # Returns
/// `Ok((outputs, states))` where `outputs[i]` contains `[lower, upper, signal]`
/// for asset `i` and `states[i]` is the final [`IndicatorState`] for asset `i`.
/// Returns `Err(IndicatorError)` if any input is too short or options are invalid.
pub(crate) fn indicator_by_assets<const N: usize>(
    inputs: &[&[&[f64]; INPUTS]; N], //stock[ fields [ field [f64] ] ]
    options: &[f64; OPTIONS],
    _optional_outputs: Option<&[bool]>,
) -> Result<(Vec<Vec<Vec<f64>>>, Vec<IndicatorState>), IndicatorError> {
    validate_options(options)?;
    validate_inputs::<INPUTS>(inputs, Breakout::min_data(options))?;
    let periods = (options[0] as usize, options[0] as usize - 1);

    let mut road_train = PrimeMover::<N, State<Warm>>::new();
    let mut output_buffers = Vec::with_capacity(N);

    for i in 0..N {
        let [high, low, close] = *inputs[i];
        let asset_inputs = vec![high, low, close];

        let (lower_line, upper_line, signal_line) = {
            let len = inputs[i][0].len();
            let capacity = Breakout::output_length(len, options);
            (
                crate::uninit_vec!(f64, capacity),
                crate::uninit_vec!(f64, capacity),
                crate::uninit_vec!(f64, capacity),
            )
        };

        let state = State::init_state(high, low, periods.1);

        let mut output_buffer = vec![lower_line, upper_line, signal_line];

        let mut asset_outputs = Vec::with_capacity(output_buffer.len());

        for j in 0..output_buffer.len() {
            unsafe {
                let output_buffer = &mut output_buffer[j];
                asset_outputs.push(std::slice::from_raw_parts_mut(
                    output_buffer.as_mut_ptr(),
                    output_buffer.len(),
                ));
            }
        }

        // First output is bar `period`, and every epoch needs `period` bars ahead of it.
        road_train.add_asset(Asset::new(
            asset_inputs,
            asset_outputs,
            i,
            periods.0,
            periods.0,
            state,
            None,
        ));
        output_buffers.push(output_buffer);
    }

    let mut driver = BreakoutDriver {
        period: periods.0,
        look_back: periods.1,
    };
    let states_vec = road_train.drive(&mut driver);
    let mut states = Vec::with_capacity(N);
    for (i, state) in states_vec.into_iter().enumerate() {
        states.push(IndicatorState::new(
            state,
            inputs[i][0],
            inputs[i][1],
            periods,
        ));
    }
    Ok((output_buffers, states))
}
