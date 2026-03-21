# -----------------------------
# Benchmark helper (paste into app.py)
# -----------------------------
import time
import math

def run_benchmark_ui():
    st.subheader("Benchmark: one-symbol grid timing")
    bench_symbol = st.text_input("Representative symbol for benchmark", value="SPY")
    n_symbols_total = st.number_input("Project total symbols", value=1500, min_value=1, step=1)
    worker_options_text = st.text_input("Worker counts to project (comma-separated)", value="1,4,8,16")
    run_bench = st.button("Run benchmark for one symbol")

    if run_bench:
        base_path = base_cache_path(bench_symbol)
        if not os.path.exists(base_path):
            st.error(f"Base features not found for {bench_symbol}. Run ensure_base_features first.")
            return

        base_df = load_base_features(bench_symbol)
        if base_df.empty or len(base_df) < 260:
            st.error(f"Base features for {bench_symbol} are empty or too short ({len(base_df)} rows).")
            return

        # Build combo list (must match your grid)
        combo_list = []
        for tsi in TSI_OPTIONS:
            for cci in CCI_OPTIONS:
                for adx in ADX_OPTIONS:
                    for thr in TSI_THRESHOLDS:
                        cci_states = ["down_1d","down_2d"]
                        adx_states = ["flat_or_down","down_1d","any"]
                        for cci_state in cci_states:
                            for adx_state in adx_states:
                                combo_list.append((tsi, cci, adx, thr, cci_state, adx_state))

        n_combos = len(combo_list)
        st.write(f"Running {n_combos} combos for {bench_symbol} (rows: {len(base_df)}) — this may take a minute.")

        # Warm up
        try:
            _ = backtest_combo(base_df, TSI_OPTIONS[0], CCI_OPTIONS[0], ADX_OPTIONS[0], TSI_THRESHOLDS[0],
                               "down_1d", "flat_or_down", "bearish", False, False)
        except Exception:
            st.warning("Warmup call to backtest_combo failed; benchmark may error if backtest_combo is not in scope.")

        start = time.time()
        progress = st.progress(0)
        for i, params in enumerate(combo_list, start=1):
            tsi_params, cci_len, adx_len, tsi_thr, cci_state, adx_state = params
            _ = backtest_combo(
                base_df=base_df,
                tsi_params=tsi_params,
                cci_len=cci_len,
                adx_len=adx_len,
                tsi_pct_threshold=tsi_thr,
                cci_state=cci_state,
                adx_state=adx_state,
                direction="bearish",
                require_rejection=False,
                require_extension=False,
            )
            if i % 50 == 0 or i == n_combos:
                progress.progress(i / n_combos)

        total_seconds = time.time() - start
        sec_per_symbol = total_seconds
        sec_per_combo = total_seconds / n_combos

        st.success("Benchmark complete")
        st.write(f"**Total time for {n_combos} combos on {bench_symbol}:** {total_seconds:.1f} seconds")
        st.write(f"**Average time per combo:** {sec_per_combo:.4f} seconds")
        st.write(f"**Average time per symbol (this run):** {sec_per_symbol:.1f} seconds")

        # projections
        try:
            worker_options = [int(x.strip()) for x in worker_options_text.split(",") if x.strip()]
        except Exception:
            worker_options = [1,4,8,16]

        st.write("Projected wall time for different worker counts (perfect scaling assumption):")
        rows = []
        for w in worker_options:
            proj_sec = (n_symbols_total * sec_per_symbol) / max(1, w)
            rows.append({"workers": w, "minutes": proj_sec / 60.0, "hours": proj_sec / 3600.0})
        st.table(pd.DataFrame(rows))

        st.info("Notes: projections assume base features are cached and perfect scaling. Add ~1–3s per symbol for downloads if cache is missing.")
        progress.empty()

# call the UI function somewhere appropriate in your app (e.g., in the sidebar or a diagnostics tab)
with st.expander("Diagnostics / Benchmark"):
    run_benchmark_ui()


