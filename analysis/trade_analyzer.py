import pandas as pd


def evaluate_trades(df):
    """
    Evaluate SMMA crossover trades.

    BUY:
        Entry = BUY crossover
        Exit  = next SELL crossover

    SELL:
        Entry = SELL crossover
        Exit  = next BUY crossover
    """

    trades = []

    current_trade = None

    for _, row in df.iterrows():

        # -------------------------------------------------
        # BUY SIGNAL
        # -------------------------------------------------
        if row["BUY_SIGNAL"]:

            # Close existing SELL trade
            if current_trade is not None and current_trade["signal"] == "SELL":

                exit_ltp = row["close"]

                pnl = current_trade["entry_ltp"] - exit_ltp

                trades.append({
                    **current_trade,
                    "exit_timestamp": row["timestamp"],
                    "exit_ltp": exit_ltp,
                    "pnl": pnl,
                    "profitable": int(pnl > 0)
                })

                current_trade = None

            # Open BUY trade
            if current_trade is None:

                current_trade = {
                    "signal": "BUY",
                    "entry_timestamp": row["timestamp"],
                    "entry_ltp": row["close"],
                    "entry_smma20": row["SMMA20"],
                    "entry_smma120": row["SMMA120"]
                }

        # -------------------------------------------------
        # SELL SIGNAL
        # -------------------------------------------------
        elif row["SELL_SIGNAL"]:

            # Close existing BUY trade
            if current_trade is not None and current_trade["signal"] == "BUY":

                exit_ltp = row["close"]

                pnl = exit_ltp - current_trade["entry_ltp"]

                trades.append({
                    **current_trade,
                    "exit_timestamp": row["timestamp"],
                    "exit_ltp": exit_ltp,
                    "pnl": pnl,
                    "profitable": int(pnl > 0)
                })

                current_trade = None

            # Open SELL trade
            if current_trade is None:

                current_trade = {
                    "signal": "SELL",
                    "entry_timestamp": row["timestamp"],
                    "entry_ltp": row["close"],
                    "entry_smma20": row["SMMA20"],
                    "entry_smma120": row["SMMA120"]
                }

    return pd.DataFrame(trades)