#[cfg(test)]
mod tests {
    use tulip_rs::indicators::breakout::{
        Breakout, Indicator, IndicatorByOptions, IndicatorState, TIndicatorState, INPUTS,
    };
    use tulip_rs::indicators::donchianchannel::DonchianChannel;
    use tulip_test::database::{get_all_stock_data, init_database_data};

    const CHUNK_SIZE: usize = 100;

    const HIGH: [f64; 15] = [
        82.15, 81.89, 83.03, 83.30, 83.85, 83.90, 83.33, 84.30, 84.84, 85.00, 85.90, 86.58, 86.98,
        88.00, 87.87,
    ];
    const LOW: [f64; 15] = [
        81.29, 80.64, 81.31, 82.65, 83.07, 83.11, 82.49, 82.30, 84.15, 84.11, 84.03, 85.39, 85.76,
        87.17, 87.01,
    ];
    const CLOSE: [f64; 15] = [
        81.59, 81.06, 82.87, 83.00, 83.61, 83.15, 82.84, 83.99, 84.55, 84.36, 85.53, 86.54, 86.89,
        87.77, 87.29,
    ];

    // Options for Breakout: [period]
    const OPTIONS_LIST: [[f64; 1]; 4] = [[5.0], [14.0], [20.0], [50.0]];

    /// Straightforward reference: for each bar `i >= period`, the highest high and lowest low
    /// of bars `i - period ..= i - 1`, and the close classified strictly against them.
    fn naive(high: &[f64], low: &[f64], close: &[f64], period: usize) -> [Vec<f64>; 3] {
        let (mut lower, mut upper, mut signal) = (Vec::new(), Vec::new(), Vec::new());
        for i in period..high.len() {
            let hi = high[i - period..i].iter().cloned().fold(f64::MIN, f64::max);
            let lo = low[i - period..i].iter().cloned().fold(f64::MAX, f64::min);
            lower.push(lo);
            upper.push(hi);
            signal.push(if close[i] > hi {
                1.0
            } else if close[i] < lo {
                -1.0
            } else {
                0.0
            });
        }
        [lower, upper, signal]
    }

    /// A deterministic random walk with varied ranges, so breakouts in both directions,
    /// ties and long quiet stretches all occur. `seed` selects the series.
    fn walk(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let (mut high, mut low, mut close) = (Vec::new(), Vec::new(), Vec::new());
        let mut price = 50.0 + 50.0 * next();
        for _ in 0..n {
            price = (price * (1.0 + (next() - 0.5) * 0.06)).max(0.5);
            // Rounding to cents makes exact ties with prior highs/lows routine.
            let h = ((price * (1.0 + next() * 0.02)) * 100.0).round() / 100.0;
            let l = ((price * (1.0 - next() * 0.02)) * 100.0).round() / 100.0;
            let c = ((l + (h - l) * next()) * 100.0).round() / 100.0;
            high.push(h);
            low.push(l);
            close.push(c);
        }
        (high, low, close)
    }

    fn expand_inputs() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (mut h, mut l, mut c) = (HIGH.to_vec(), LOW.to_vec(), CLOSE.to_vec());
        for _ in 0..3 {
            h.extend_from_slice(&HIGH);
            l.extend_from_slice(&LOW);
            c.extend_from_slice(&CLOSE);
        }
        (h, l, c) // 60 bars — enough for period up to 50
    }

    fn get_arrays(
        stock_data: &[tulip_test::database::EodData],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let high = stock_data.iter().map(|d| d.high).collect();
        let low = stock_data.iter().map(|d| d.low).collect();
        let close = stock_data.iter().map(|d| d.close).collect();
        (high, low, close)
    }

    fn assert_lines_eq(got: &[Vec<f64>], want: &[Vec<f64>], what: &str) {
        let names = ["lower", "upper", "signal"];
        assert_eq!(got.len(), 3, "{what}: expected 3 outputs");
        for k in 0..3 {
            assert_eq!(
                got[k].len(),
                want[k].len(),
                "{what}: {} length mismatch",
                names[k]
            );
            for (i, (&g, &w)) in got[k].iter().zip(want[k].iter()).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "{what}: {} mismatch at index {i}: got={g}, want={w}",
                    names[k]
                );
            }
        }
    }

    /// Runs `indicator` on the first `seed_len` bars, then `batch_indicator` over the rest in
    /// `CHUNK_SIZE` pieces, and returns the concatenated outputs.
    fn streamed(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        options: &[f64; 1],
        seed_len: usize,
    ) -> Vec<Vec<f64>> {
        let (first, mut state) =
            Breakout::indicator(&[&high[..seed_len], &low[..seed_len], &close[..seed_len]], options, None)
                .expect("Breakout indicator failed on seed");
        let mut out = first;
        continue_batches(&mut state, &mut out, high, low, close, seed_len);
        out
    }

    fn continue_batches(
        state: &mut IndicatorState,
        out: &mut [Vec<f64>],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        from: usize,
    ) {
        let mut start = from;
        while start < high.len() {
            let end = (start + CHUNK_SIZE).min(high.len());
            let chunk = state
                .batch_indicator(&[&high[start..end], &low[start..end], &close[start..end]], None)
                .expect("Breakout batch_indicator failed");
            for k in 0..out.len() {
                out[k].extend_from_slice(&chunk[k]);
            }
            start = end;
        }
    }

    // -------------------------------------------------------------------------
    // Definition
    // -------------------------------------------------------------------------

    /// Hand-checked values on the canonical 15 bars with period 5.
    #[test]
    fn test_breakout_hand_checked() {
        let (outputs, _) = Breakout::indicator(&[&HIGH, &LOW, &CLOSE], &[5.0], None).unwrap();
        // First output is bar 5: channel over bars 0..=4.
        assert_eq!(outputs[0].len(), 10);
        assert_eq!(outputs[0][0], 80.64); // min(low[0..5])
        assert_eq!(outputs[1][0], 83.85); // max(high[0..5])
        assert_eq!(outputs[2][0], 0.0); // close 83.15 inside
        // Bar 7: channel over bars 2..=6 has upper 83.90; close 83.99 breaks above.
        assert_eq!(outputs[1][2], 83.90);
        assert_eq!(outputs[2][2], 1.0);
        // Bar 6: close 82.84 is inside 80.64..83.90.
        assert_eq!(outputs[2][1], 0.0);
    }

    /// A close exactly equal to the prior high or low is not a breakout.
    #[test]
    fn test_breakout_ties_are_not_breakouts() {
        let high = [10.0, 11.0, 12.0, 12.0, 12.0];
        let low = [9.0, 9.0, 10.0, 10.0, 8.0];
        let close = [9.5, 10.5, 11.0, 12.0, 9.0];
        let (o, _) = Breakout::indicator(&[&high, &low, &close], &[3.0], None).unwrap();
        // Bar 3: prior high 12.0, close 12.0 -> 0. Bar 4: prior low 9.0, close 9.0 -> 0.
        assert_eq!(o[2], vec![0.0, 0.0]);
        let close = [9.5, 10.5, 11.0, 12.01, 8.99];
        let (o, _) = Breakout::indicator(&[&high, &low, &close], &[3.0], None).unwrap();
        assert_eq!(o[2], vec![1.0, -1.0]);
    }

    /// Period 1 compares each close with the previous bar's high and low.
    #[test]
    fn test_breakout_period_one() {
        let (o, _) = Breakout::indicator(&[&HIGH, &LOW, &CLOSE], &[1.0], None).unwrap();
        assert_lines_eq(&o, &naive(&HIGH, &LOW, &CLOSE, 1), "period 1");
    }

    /// `min_data` is `period + 1`; one bar fewer is refused, exactly that many gives one output.
    #[test]
    fn test_breakout_min_data() {
        assert_eq!(Breakout::min_data(&[5.0]), 6);
        assert!(Breakout::indicator(&[&HIGH[..5], &LOW[..5], &CLOSE[..5]], &[5.0], None).is_err());
        let (o, _) = Breakout::indicator(&[&HIGH[..6], &LOW[..6], &CLOSE[..6]], &[5.0], None).unwrap();
        assert_eq!(o[0].len(), 1);
        assert!(Breakout::indicator(&[&HIGH, &LOW, &CLOSE[..14]], &[5.0], None).is_err());
        assert!(Breakout::indicator(&[&HIGH, &LOW, &CLOSE], &[0.0], None).is_err());
    }

    /// The channel is Donchian's, one bar later: Breakout's output `j` (bar `period + j`)
    /// carries Donchian's output `j` (bar `period - 1 + j`). Bit-for-bit, since both use the
    /// same min/max states.
    #[test]
    fn test_breakout_is_donchian_shifted_one_bar() {
        for seed in 0..4 {
            let (high, low, close) = walk(3000, seed);
            for options in OPTIONS_LIST {
                let (b, _) = Breakout::indicator(&[&high, &low, &close], &options, None).unwrap();
                let (d, _) = DonchianChannel::indicator(&[&high, &low], &options, None).unwrap();
                assert_eq!(b[0].len() + 1, d[0].len());
                for j in 0..b[0].len() {
                    assert_eq!(b[0][j].to_bits(), d[0][j].to_bits(), "lower vs donchian at {j}");
                    assert_eq!(b[1][j].to_bits(), d[2][j].to_bits(), "upper vs donchian at {j}");
                }
            }
        }
    }

    #[test]
    fn test_breakout_matches_naive_reference() {
        let (high, low, close) = expand_inputs();
        for options in OPTIONS_LIST {
            let (o, _) = Breakout::indicator(&[&high, &low, &close], &options, None).unwrap();
            assert_lines_eq(&o, &naive(&high, &low, &close, options[0] as usize), "expanded");
        }
        for seed in 0..4 {
            let (high, low, close) = walk(3000, seed);
            for options in OPTIONS_LIST {
                let (o, _) = Breakout::indicator(&[&high, &low, &close], &options, None).unwrap();
                let want = naive(&high, &low, &close, options[0] as usize);
                assert_lines_eq(&o, &want, &format!("walk seed {seed} options {options:?}"));
                // The walk must actually exercise every outcome.
                assert!(want[2].contains(&1.0) && want[2].contains(&-1.0) && want[2].contains(&0.0));
            }
        }
    }

    #[test]
    fn test_breakout_state_bincode_roundtrip() {
        use bincode::config::standard;
        use bincode::serde::{decode_from_slice, encode_to_vec};

        let (high, low, close) = expand_inputs();
        let mid = 50;
        let options = [14.0];
        let first: [&[f64]; INPUTS] = [&high[..mid], &low[..mid], &close[..mid]];
        let second: [&[f64]; INPUTS] = [&high[mid..], &low[mid..], &close[mid..]];

        let (_rows, mut original) = Breakout::indicator(&first, &options, None).unwrap();
        let cfg = standard();
        let bytes = encode_to_vec(&original, cfg).expect("serialize");
        let (mut restored, consumed): (IndicatorState, usize) =
            decode_from_slice(&bytes, cfg).expect("deserialize");
        assert_eq!(consumed, bytes.len());

        let a = original.batch_indicator(&second, None).unwrap();
        let b = restored.batch_indicator(&second, None).unwrap();
        assert_lines_eq(&a, &b, "bincode roundtrip");
    }

    // -------------------------------------------------------------------------
    // Streaming: batch_indicator vs full indicator
    // -------------------------------------------------------------------------

    #[test]
    fn test_breakout_state_continuity() {
        for seed in 0..4 {
            let (high, low, close) = walk(2500, seed);
            for options in OPTIONS_LIST {
                let (full, _) = Breakout::indicator(&[&high, &low, &close], &options, None).unwrap();
                // Seed with the minimum, and with a longer prefix.
                for seed_len in [Breakout::min_data(&options), 777] {
                    let got = streamed(&high, &low, &close, &options, seed_len);
                    assert_lines_eq(&got, &full, &format!("stream seed_len {seed_len} {options:?}"));
                }
                // One bar at a time.
                let m = Breakout::min_data(&options);
                let (mut out, mut state) =
                    Breakout::indicator(&[&high[..m], &low[..m], &close[..m]], &options, None).unwrap();
                for i in m..high.len() {
                    let c = state
                        .batch_indicator(&[&high[i..i + 1], &low[i..i + 1], &close[i..i + 1]], None)
                        .unwrap();
                    for k in 0..3 {
                        out[k].extend_from_slice(&c[k]);
                    }
                }
                assert_lines_eq(&out, &full, &format!("bar-by-bar {options:?}"));
            }
        }
    }

    #[test]
    fn test_breakout_database_state() {
        init_database_data();
        let data = get_all_stock_data().unwrap();
        for (stock_symbol, stock_data) in data {
            let (high, low, close) = get_arrays(stock_data);
            for options in OPTIONS_LIST {
                let (full, _) = Breakout::indicator(&[&high, &low, &close], &options, None).unwrap();
                assert_lines_eq(
                    &full,
                    &naive(&high, &low, &close, options[0] as usize),
                    &format!("database naive {stock_symbol} {options:?}"),
                );
                let seed_len = Breakout::min_data(&options).max(CHUNK_SIZE);
                let got = streamed(&high, &low, &close, &options, seed_len);
                assert_lines_eq(&got, &full, &format!("database stream {stock_symbol} {options:?}"));
            }
        }
    }

    // -------------------------------------------------------------------------
    // SIMD by-assets
    // -------------------------------------------------------------------------

    /// Assets of DIFFERENT lengths, so the scheduler picks them up at different points.
    #[test]
    fn test_breakout_simd_by_assets_vs_regular() {
        let series: Vec<_> = [3000, 2400, 2999, 1200]
            .iter()
            .enumerate()
            .map(|(k, &n)| walk(n, 10 + k as u64))
            .collect();
        for options in OPTIONS_LIST {
            let a: Vec<[&[f64]; 3]> = series
                .iter()
                .map(|(h, l, c)| [h.as_slice(), l.as_slice(), c.as_slice()])
                .collect();
            let inputs_4: [&[&[f64]; 3]; 4] = [&a[0], &a[1], &a[2], &a[3]];
            let (simd, _) = Breakout::indicator_by_assets::<4>(&inputs_4, &options, None)
                .expect("SIMD by assets failed");
            for (k, (h, l, c)) in series.iter().enumerate() {
                let (want, _) = Breakout::indicator(&[h, l, c], &options, None).unwrap();
                assert_lines_eq(&simd[k], &want, &format!("by_assets asset {k} {options:?}"));
            }
        }
    }

    #[test]
    fn test_breakout_simd_by_assets_vs_regular_database() {
        init_database_data();
        let data = get_all_stock_data().unwrap();
        let stock: Vec<_> = data.iter().take(4).map(|(s, eod)| (s.clone(), get_arrays(eod))).collect();
        for options in OPTIONS_LIST {
            let a: Vec<[&[f64]; 3]> = stock
                .iter()
                .map(|(_, (h, l, c))| [h.as_slice(), l.as_slice(), c.as_slice()])
                .collect();
            let inputs_4: [&[&[f64]; 3]; 4] = [&a[0], &a[1], &a[2], &a[3]];
            let (simd, _) = Breakout::indicator_by_assets::<4>(&inputs_4, &options, None).unwrap();
            for (k, (sym, (h, l, c))) in stock.iter().enumerate() {
                let (want, _) = Breakout::indicator(&[h, l, c], &options, None).unwrap();
                assert_lines_eq(&simd[k], &want, &format!("by_assets {sym} {options:?}"));
            }
        }
    }

    /// First 1000 bars via SIMD by-assets, the rest via each asset's state.
    #[test]
    fn test_breakout_simd_by_assets_state_continuity() {
        const FIRST_CHUNK: usize = 1000;
        let series: Vec<_> = [3000, 2400, 2999, 1200]
            .iter()
            .enumerate()
            .map(|(k, &n)| walk(n, 20 + k as u64))
            .collect();
        for options in OPTIONS_LIST {
            let a: Vec<[&[f64]; 3]> = series
                .iter()
                .map(|(h, l, c)| [&h[..FIRST_CHUNK], &l[..FIRST_CHUNK], &c[..FIRST_CHUNK]])
                .collect();
            let inputs_4: [&[&[f64]; 3]; 4] = [&a[0], &a[1], &a[2], &a[3]];
            let (first, mut states) =
                Breakout::indicator_by_assets::<4>(&inputs_4, &options, None).unwrap();
            for (k, (h, l, c)) in series.iter().enumerate() {
                let mut out = first[k].clone();
                continue_batches(&mut states[k], &mut out, h, l, c, FIRST_CHUNK);
                let (want, _) = Breakout::indicator(&[h, l, c], &options, None).unwrap();
                assert_lines_eq(&out, &want, &format!("by_assets continuity {k} {options:?}"));
            }
        }
    }

    // -------------------------------------------------------------------------
    // SIMD by-options
    // -------------------------------------------------------------------------

    #[test]
    fn test_breakout_simd_by_options_vs_regular() {
        let options_4 = [&OPTIONS_LIST[0], &OPTIONS_LIST[1], &OPTIONS_LIST[2], &OPTIONS_LIST[3]];
        for seed in 0..4 {
            let (h, l, c) = walk(3000, 30 + seed);
            let (simd, _) =
                Breakout::indicator_by_options::<4>(&[&h, &l, &c], &options_4, None).unwrap();
            for (k, options) in OPTIONS_LIST.iter().enumerate() {
                let (want, _) = Breakout::indicator(&[&h, &l, &c], options, None).unwrap();
                assert_lines_eq(&simd[k], &want, &format!("by_options seed {seed} {options:?}"));
            }
        }
    }

    #[test]
    fn test_breakout_simd_by_options_vs_regular_database() {
        init_database_data();
        let data = get_all_stock_data().unwrap();
        let options_4 = [&OPTIONS_LIST[0], &OPTIONS_LIST[1], &OPTIONS_LIST[2], &OPTIONS_LIST[3]];
        for (sym, stock_data) in data {
            let (h, l, c) = get_arrays(stock_data);
            let (simd, _) =
                Breakout::indicator_by_options::<4>(&[&h, &l, &c], &options_4, None).unwrap();
            for (k, options) in OPTIONS_LIST.iter().enumerate() {
                let (want, _) = Breakout::indicator(&[&h, &l, &c], options, None).unwrap();
                assert_lines_eq(&simd[k], &want, &format!("by_options {sym} {options:?}"));
            }
        }
    }

    /// First 1000 bars via SIMD by-options, the rest via each option set's state.
    #[test]
    fn test_breakout_simd_by_options_state_continuity() {
        const FIRST_CHUNK: usize = 1000;
        let options_4 = [&OPTIONS_LIST[0], &OPTIONS_LIST[1], &OPTIONS_LIST[2], &OPTIONS_LIST[3]];
        let (h, l, c) = walk(3000, 40);
        let (first, mut states) = Breakout::indicator_by_options::<4>(
            &[&h[..FIRST_CHUNK], &l[..FIRST_CHUNK], &c[..FIRST_CHUNK]],
            &options_4,
            None,
        )
        .unwrap();
        for (k, options) in OPTIONS_LIST.iter().enumerate() {
            let mut out = first[k].clone();
            continue_batches(&mut states[k], &mut out, &h, &l, &c, FIRST_CHUNK);
            let (want, _) = Breakout::indicator(&[&h, &l, &c], options, None).unwrap();
            assert_lines_eq(&out, &want, &format!("by_options continuity {options:?}"));
        }
    }
}
