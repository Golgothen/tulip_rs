use crate::indicators::breakout::State;
#[cfg(feature = "simd_assets")]
pub(crate) use crate::indicators::simd_indicators::by_asset::breakout::indicator_by_assets;

#[cfg(feature = "simd_options")]
pub(crate) use crate::indicators::simd_indicators::by_option::breakout::indicator_by_options;

pub use crate::indicator_types::{TSimdState, TState};
use crate::types::Warm;
use crate::indicators::simd_indicators::simd_types::{F64Constants, UsizeConstants};
use std::simd::{cmp::SimdPartialOrd, Select, Simd};

/// SIMD form of [`crate::indicators::breakout::calc_signal`]: `+1` where the close is
/// strictly above `upper`, `-1` where strictly below `lower`, `0` elsewhere (NaN included).
#[inline(always)]
pub(crate) fn calc_signal_simd<const N: usize>(
    close: Simd<f64, N>,
    lower: Simd<f64, N>,
    upper: Simd<f64, N>,
) -> Simd<f64, N> {
    let above = close.simd_gt(upper);
    let below = close.simd_lt(lower);
    above.select(
        F64Constants::ONE,
        below.select(F64Constants::NEG_ONE, F64Constants::ZERO),
    )
}

/// SIMD-parallel state for the Breakout indicator across `N` assets.
/// Wraps the min/max ring-buffer SIMD states used by Donchian Channel, one per lane.
pub mod assets {
    use super::*;
    use crate::indicators::simd_indicators::{
        max_simd::assets::SimdState as MaxSimdState, min_simd::assets::SimdState as MinSimdState,
    };

    pub struct SimdState<const N: usize> {
        min_state: MinSimdState<N>,
        max_state: MaxSimdState<N>,
    }
    impl<const N: usize> TSimdState for SimdState<N> {
        type ScalarState = State<Warm>;
        crate::simd_state_impl!(
             sub: [(min_state: MinSimdState<N>), (max_state: MaxSimdState<N>)],
             scalar: []
        );
    }

    impl<const N: usize> TState for SimdState<N> {
        /// `(high_ptrs, low_ptrs, close, i, look_back)` — `i` is the bar being classified;
        /// the channel is evaluated at `i - 1`.
        type Inputs<'a> = ([*const f64; N], [*const f64; N], Simd<f64, N>, usize, usize);
        /// `(lower, upper, signal)`.
        type Outputs = (Simd<f64, N>, Simd<f64, N>, Simd<f64, N>);

        #[inline(always)]
        fn calc(
            &mut self,
            (high_ptrs, low_ptrs, close, i, look_back): Self::Inputs<'_>,
        ) -> Self::Outputs {
            let (lower, _) = self.min_state.calc((low_ptrs, i - 1, look_back));
            let (upper, _) = self.max_state.calc((high_ptrs, i - 1, look_back));

            (lower, upper, calc_signal_simd(close, lower, upper))
        }
    }
}

/// SIMD-parallel state for the Breakout indicator across `N` option sets on one asset.
pub mod options {
    use super::*;
    use crate::indicators::simd_indicators::{
        max_simd::options::SimdState as MaxSimdState, min_simd::options::SimdState as MinSimdState,
    };
    pub struct SimdState<const N: usize> {
        min_state: MinSimdState<N>,
        max_state: MaxSimdState<N>,
    }
    impl<const N: usize> TSimdState for SimdState<N> {
        type ScalarState = State<Warm>;
        crate::simd_state_impl!(
             sub: [(min_state: MinSimdState<N>), (max_state: MaxSimdState<N>)],
             scalar: []
        );
    }

    impl<const N: usize> TState for SimdState<N> {
        /// `(high_ptrs, low_ptrs, close, i, look_back)` with per-lane `i` and `look_back`;
        /// `i` is the bar being classified and the channel is evaluated at `i - 1`.
        type Inputs<'a> = (
            [*const f64; N],
            [*const f64; N],
            Simd<f64, N>,
            Simd<usize, N>,
            Simd<usize, N>,
        );
        /// `(lower, upper, signal)`.
        type Outputs = (Simd<f64, N>, Simd<f64, N>, Simd<f64, N>);

        #[inline(always)]
        fn calc(
            &mut self,
            (high_ptrs, low_ptrs, close, i, look_back): Self::Inputs<'_>,
        ) -> Self::Outputs {
            let prev = i - UsizeConstants::ONE;
            let (lower, _) = self.min_state.calc((low_ptrs, prev, look_back));
            let (upper, _) = self.max_state.calc((high_ptrs, prev, look_back));

            (lower, upper, calc_signal_simd(close, lower, upper))
        }
    }
}
