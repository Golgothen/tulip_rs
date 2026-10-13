use crate::common_simd::options::{validate_inputs, validate_options};
use crate::indicators::breakout::{Breakout, Indicator, IndicatorState, State, INPUTS, OPTIONS};
use crate::indicators::simd_indicators::breakout_simd::{options::SimdState, TSimdState, TState};
use crate::indicators::simd_indicators::road_train::{Asset, Driver, PrimeMover};
use crate::types::{IndicatorError, Warm};
use std::simd::Simd;

/// SIMD driver for the Breakout indicator, processing `N` option-set lanes per scheduling epoch.
/// Each lane's option is its `period`.
struct BreakoutDriver;

impl Driver<State<Warm>, usize> for BreakoutDriver {
    /// Processes one epoch of output bars for `N` option-set lanes simultaneously using SIMD.
    ///
    /// Each lane's slices start `period` bars before its first bar to classify; lanes advance
    /// together, one bar per step, each against the channel ending one bar earlier.
    fn next_run<const N: usize>(
        &mut self,
        inputs: Vec<Vec<&[f64]>>,
        mut outputs: Vec<Vec<&mut [f64]>>,
        mut states: Vec<&mut State<Warm>>,
        options: Vec<Option<&usize>>,
    ) {
        let len = outputs[0][0].len();

        let (look_back, mut i_simd) = {
            let mut look_back = [0; N];
            let mut i_array = [0; N];
            for (lane, option) in options.iter().enumerate() {
                if let Some(&period) = option {
                    look_back[lane] = period - 1;
                    i_array[lane] = period;
                }
            }
            (Simd::from_array(look_back), Simd::from_array(i_array))
        };

        let (lower_line_ptr, upper_line_ptr, signal_line_ptr) =
            crate::extract_output_ptrs!(outputs, N, lower, upper, signal);

        let (high_ptrs, low_ptrs, close_ptrs) =
            crate::extract_input_ptrs!(inputs, N, high_ptrs, low_ptrs, close_ptrs);

        let mut state = SimdState::<N>::from_states(&mut states);
        let one_splat = Simd::splat(1);

        for j in 0..len {
            let close: Simd<f64, N> =
                crate::extract_simd_inputs_at_index_array!(i_simd, N, c @ close_ptrs);
            let (lower, upper, signal) =
                state.calc((high_ptrs, low_ptrs, close, i_simd, look_back));

            crate::write_simd_at_indices!(N, j,
                lower_line_ptr => lower,
                upper_line_ptr => upper,
                signal_line_ptr => signal
            );

            i_simd += one_splat;
        }
        state.write_states(&mut states);
    }
}

/// Calculates the Breakout indicator on a single asset with `N` different option sets
/// simultaneously using SIMD parallelism.
///
/// # Arguments
/// * `inputs` - The single asset's price series: `[high, low, close]`.
/// * `options` - An array of `N` option sets, one per SIMD lane: `[period]`.
/// * `_optional_outputs` - Unused; pass `None`.
///
/// # Returns
/// `Ok((outputs, states))` where `outputs[i]` contains `[lower, upper, signal]`
/// and `states[i]` is the final [`IndicatorState`] for option set `i`.
/// Returns `Err(IndicatorError)` if inputs are too short or options are invalid.
pub(crate) fn indicator_by_options<const N: usize>(
    inputs: &[&[f64]; INPUTS],
    options: &[&[f64; OPTIONS]; N],
    _optional_outputs: Option<&[bool]>,
) -> Result<(Vec<Vec<Vec<f64>>>, Vec<IndicatorState>), IndicatorError> {
    validate_options(options, None)?;
    validate_inputs::<OPTIONS>(inputs, options, Breakout::min_data)?;
    let periods: [(usize, usize); N] = std::array::from_fn(|i| {
        let period = options[i][0] as usize;
        (period, period - 1)
    });
    let mut road_train = PrimeMover::<N, State<Warm>, usize>::new();
    let mut output_buffers = Vec::with_capacity(N);

    let [high, low, close] = *inputs;
    for i in 0..N {
        let asset_inputs = vec![high, low, close];

        let (lower_line, upper_line, signal_line) = {
            let capacity = Breakout::output_length(high.len(), options[i]);
            (
                crate::uninit_vec!(f64, capacity),
                crate::uninit_vec!(f64, capacity),
                crate::uninit_vec!(f64, capacity),
            )
        };
        let state = State::init_state(high, low, periods[i].1);

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
            periods[i].0,
            periods[i].0,
            state,
            Some(&periods[i].0),
        ));
        output_buffers.push(output_buffer);
    }

    let mut driver = BreakoutDriver;

    let states_vec = road_train.drive(&mut driver);
    let mut states = Vec::with_capacity(N);
    for (state, &periods) in states_vec.into_iter().zip(periods.iter()) {
        states.push(IndicatorState::new(state, high, low, periods));
    }
    Ok((output_buffers, states))
}
