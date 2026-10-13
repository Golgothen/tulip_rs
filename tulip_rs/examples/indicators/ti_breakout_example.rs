use tulip_rs::indicators::breakout::{Breakout, Indicator, IndicatorByOptions, TIndicatorState};

fn main() {
    let high = [
        82.15, 81.89, 83.03, 83.30, 83.85, 83.90, 83.33, 84.30, 84.84, 85.00, 85.90, 86.58, 86.98,
        88.00, 87.87,
    ]; // High prices (15 bars)
    let low = [
        81.29, 80.64, 81.31, 82.65, 83.07, 83.11, 82.49, 82.30, 84.15, 84.11, 84.03, 85.39, 85.76,
        87.17, 87.01,
    ]; // Low prices (15 bars)
    let close = [
        81.59, 81.06, 82.87, 83.00, 83.61, 83.15, 82.84, 83.99, 84.55, 84.36, 85.53, 86.54, 86.89,
        87.77, 87.29,
    ]; // Close prices (15 bars)
    let options = [5.0]; // Period=5: each close is measured against the 5 bars before it

    let inputs = [high.as_slice(), low.as_slice(), close.as_slice()];

    /////////////////////// Full run — channel of the prior bars, and the signal ///////////////////////
    let (outputs, _) = match Breakout::indicator(&inputs, &options, None) {
        Ok(result) => result,
        Err(e) => panic!("Error: {}", e),
    };
    println!("Full Lower  (prior lowest low):   {:?}", outputs[0]);
    println!("Full Upper  (prior highest high): {:?}", outputs[1]);
    println!("Full Signal (+1 up, -1 down, 0):  {:?}", outputs[2]);

    /////////////////////// Partial run + batch_indicator continuation ///////////////////////
    let n = high.len() - 5;
    let inputs2 = [&high[..n], &low[..n], &close[..n]];

    let (outputs2, mut state) = match Breakout::indicator(&inputs2, &options, None) {
        Ok(result) => result,
        Err(e) => panic!("Error: {}", e),
    };
    println!("\nPartial Lower:  {:?}", outputs2[0]);
    println!("Partial Upper:  {:?}", outputs2[1]);
    println!("Partial Signal: {:?}", outputs2[2]);

    let new_inputs = [&high[n..], &low[n..], &close[n..]];

    let final_outputs = match state.batch_indicator(&new_inputs, None) {
        Ok(result) => result,
        Err(e) => panic!("Error: {}", e),
    };
    println!("\nContinued Lower:  {:?}", final_outputs[0]);
    println!("Continued Upper:  {:?}", final_outputs[1]);
    println!("Continued Signal: {:?}", final_outputs[2]);

    /////////////////////// SIMD by-assets: 4 assets, same options ///////////////////////
    let scaled = |k: f64| -> [Vec<f64>; 3] {
        [
            high.iter().map(|v| v * k).collect(),
            low.iter().map(|v| v * k).collect(),
            close.iter().map(|v| v * k).collect(),
        ]
    };
    let (s1, s2, s3) = (scaled(1.2), scaled(0.9), scaled(1.05));
    let asset0: [&[f64]; 3] = [&high, &low, &close];
    let asset1: [&[f64]; 3] = [&s1[0], &s1[1], &s1[2]];
    let asset2: [&[f64]; 3] = [&s2[0], &s2[1], &s2[2]];
    let asset3: [&[f64]; 3] = [&s3[0], &s3[1], &s3[2]];
    let inputs_4: [&[&[f64]; 3]; 4] = [&asset0, &asset1, &asset2, &asset3];

    let (simd_asset_outputs, _) =
        match Breakout::indicator_by_assets::<4>(&inputs_4, &options, None) {
            Ok(result) => result,
            Err(e) => panic!("Error: {}", e),
        };
    for (i, out) in simd_asset_outputs.iter().enumerate() {
        println!("\nSIMD by-assets Signal (asset {}): {:?}", i, out[2]);
    }

    /////////////////////// SIMD by-options: 1 asset, 4 option sets ///////////////////////
    // Breakout needs period + 1 bars, so with 15 bars:
    //   period=3  → 12 output values
    //   period=5  → 10 output values
    //   period=7  → 8 output values
    //   period=10 → 5 output values
    let options_4 = [&[3.0], &[5.0], &[7.0], &[10.0]];

    let (simd_option_outputs, _) =
        match Breakout::indicator_by_options::<4>(&inputs, &options_4, None) {
            Ok(result) => result,
            Err(e) => panic!("Error: {}", e),
        };
    for (i, opts) in options_4.iter().enumerate() {
        println!(
            "\nSIMD by-options (period={}) Lower: {:?}\n  Upper: {:?}\n  Signal: {:?}",
            opts[0], simd_option_outputs[i][0], simd_option_outputs[i][1], simd_option_outputs[i][2]
        );
    }
}
