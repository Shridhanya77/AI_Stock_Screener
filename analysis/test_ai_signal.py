from analysis.ai_signal import get_ai_signal


result = get_ai_signal(
    signal="BUY",
    entry_ltp=1200,
    ltq=5,
    ltq_avg_2m=5,
    ltq_avg_5m=5,
    ltq_ratio=1.0,
    ltq_change_pct=0,
    ltq_zscore=0,
    ltp_change_pct=0,
    ltp_std_5m=0
)

print("=" * 60)
print("AI STOCK SIGNAL")
print("=" * 60)

print(f"SMMA Signal : {result['smma_signal']}")
print(f"ML Result   : {result['ml_result']}")
print(f"Confidence  : {result['confidence']}%")
print(f"Final Signal: {result['final_signal']}")