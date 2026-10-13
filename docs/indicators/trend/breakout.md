# Breakout

Signals a close beyond the Donchian channel of the **preceding** `period` bars. At each bar, `upper` is the highest high and `lower` the lowest low of the `period` bars before it — the current bar is excluded, which is what makes a breakout possible (a channel that includes the current bar can never be closed above). `signal` is `+1` when the close is strictly above `upper`, `-1` when strictly below `lower`, and `0` otherwise; a close equal to either level is not a breakout. `lower` and `upper` are the levels the signal was measured against and render as a price overlay; `signal` renders in its own pane.

Needs `period + 1` bars; the first output is for bar `period`. The channel equals [Donchian Channel](donchianchannel.md)'s `lower`/`upper` one bar later.

**Inputs:** `[high, low, close]` &nbsp;|&nbsp; **Options:** `[period]` &nbsp;|&nbsp; **Outputs:** `[lower, upper, signal]`

### Basic

=== "Rust"

    ```rust
    use tulip_rs::indicators::breakout::{Breakout, Indicator, TIndicatorState};

    let high  = vec![82.15, 81.89, 83.03, 83.30, 83.85,
                     83.90, 83.33, 84.30, 84.84, 85.00_f64];
    let low   = vec![81.29, 80.64, 81.31, 82.65, 83.07,
                     83.11, 82.49, 82.30, 84.15, 84.11_f64];
    let close = vec![81.59, 81.06, 82.87, 83.00, 83.61,
                     83.15, 82.84, 83.99, 84.55, 84.36_f64];

    let inputs = [high.as_slice(), low.as_slice(), close.as_slice()];
    let (outputs, _) = Breakout::indicator(&inputs, &[5.0], None).unwrap();
    println!("Lower:  {:?}", outputs[0]);
    println!("Upper:  {:?}", outputs[1]);
    println!("Signal: {:?}", outputs[2]); // [0.0, 0.0, 1.0, 1.0, 0.0]

    // State continuation — feed new bars without reprocessing history
    let (_, mut state) = Breakout::indicator(
        &[&high[..8], &low[..8], &close[..8]], &[5.0], None).unwrap();
    let continued = state
        .batch_indicator(&[&high[8..], &low[8..], &close[8..]], None)
        .unwrap();
    println!("Continued Signal: {:?}", continued[2]);
    ```

=== "Python"

    ```python
    import numpy as np
    import tulip_rs

    high  = np.array([82.15, 81.89, 83.03, 83.30, 83.85,
                      83.90, 83.33, 84.30, 84.84, 85.00], dtype=np.float64)
    low   = np.array([81.29, 80.64, 81.31, 82.65, 83.07,
                      83.11, 82.49, 82.30, 84.15, 84.11], dtype=np.float64)
    close = np.array([81.59, 81.06, 82.87, 83.00, 83.61,
                      83.15, 82.84, 83.99, 84.55, 84.36], dtype=np.float64)

    outputs, state = tulip_rs.indicators.breakout.indicator([high, low, close], [5.0])
    print("Lower: ", outputs[0])
    print("Upper: ", outputs[1])
    print("Signal:", outputs[2])

    # State continuation
    outputs2, state = tulip_rs.indicators.breakout.indicator([high[:8], low[:8], close[:8]], [5.0])
    continued = state.batch_indicator([high[8:], low[8:], close[8:]])
    print("Continued Signal:", continued[2])
    ```

### SIMD

=== "Rust"

    **By assets** — same period applied to 4 assets in parallel:

    ```rust
    use tulip_rs::indicators::breakout::{Breakout, Indicator};

    let h = vec![82.15, 81.89, 83.03, 83.30, 83.85, 83.90, 83.33, 84.30, 84.84, 85.00_f64];
    let l = vec![81.29, 80.64, 81.31, 82.65, 83.07, 83.11, 82.49, 82.30, 84.15, 84.11_f64];
    let c = vec![81.59, 81.06, 82.87, 83.00, 83.61, 83.15, 82.84, 83.99, 84.55, 84.36_f64];
    let asset: [&[f64]; 3] = [h.as_slice(), l.as_slice(), c.as_slice()];

    let inputs: [&[&[f64]; 3]; 4] = [&asset, &asset, &asset, &asset];
    let (results, _) = Breakout::indicator_by_assets::<4>(&inputs, &[5.0], None).unwrap();
    for (i, asset_outputs) in results.iter().enumerate() {
        println!("Asset {} Signal: {:?}", i + 1, asset_outputs[2]);
    }
    ```

    **By options** — same asset, 4 different periods in parallel:

    ```rust
    use tulip_rs::indicators::breakout::{Breakout, IndicatorByOptions};

    let opts: [&[f64; 1]; 4] = [&[2.0], &[3.0], &[4.0], &[5.0]];
    let inputs = [h.as_slice(), l.as_slice(), c.as_slice()];
    let (results, _) = Breakout::indicator_by_options::<4>(&inputs, &opts, None).unwrap();
    for (i, opt_outputs) in results.iter().enumerate() {
        println!("Period {} Signal: {:?}", opts[i][0], opt_outputs[2]);
    }
    ```

=== "Python"

    **By assets** — same period applied to N assets in parallel (must be 2, 4, 8, or 16):

    ```python
    simd_inputs = [
        [high,        low,        close],
        [high + 0.5,  low + 0.5,  close + 0.5],
        [high - 0.5,  low - 0.5,  close - 0.5],
        [high * 1.01, low * 1.01, close * 1.01],
    ]
    outputs_list, states = tulip_rs.indicators.breakout.simd_by_assets(simd_inputs, [5.0])
    for i, out in enumerate(outputs_list):
        print(f"Asset {i + 1} Signal: {out[2]}")
    ```

    **By options** — same asset, N different periods in parallel:

    ```python
    simd_options = [[2.0], [3.0], [4.0], [5.0]]
    outputs_list, states = tulip_rs.indicators.breakout.simd_by_options([high, low, close], simd_options)
    for i, out in enumerate(outputs_list):
        print(f"Period {simd_options[i][0]} Signal: {out[2]}")
    ```
