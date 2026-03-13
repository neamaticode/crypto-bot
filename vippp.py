class TelegramFormatter:
    """Formats trading signals into the Quant Analyst style for Telegram."""

    CHANNEL = "@NovaCryptoSignal"

    @staticmethod
    def _clean_symbol(raw_sym: str) -> str:
        """Convert e.g. 'DRIFT/USDT:USDT' -> '#DRIFTUSDT'."""
        return "#" + raw_sym.split(":")[0].replace("/", "")

    @staticmethod
    def _trend_label(market_trend: str) -> str:
        """Return a human-readable trend label with icon."""
        trend_lower = (market_trend or "").strip().lower()
        if trend_lower in ("up", "upward", "bullish", "trending up"):
            return "📈 Upward"
        return "📉 Downward"

    def signal(
        self,
        raw_sym: str,
        direction: str,
        entry: float,
        sl: float,
        tp1: float,
        tp2: float,
        tp3: float,
        leverage: int,
        win_prob: float,
        market_trend: str,
        catalyst: str = "",
    ) -> str:
        sym = self._clean_symbol(raw_sym)
        arrow = "📈" if direction.upper() == "LONG" else "📉"
        trend_label = self._trend_label(market_trend)
        bias = "Bullish" if direction.upper() == "LONG" else "Bearish"
        default_catalyst = (
            f"Strong MTF {bias} Confirmation & Market Structure Break."
        )
        catalyst_text = catalyst.strip() if catalyst.strip() else default_catalyst
        return (
            f"{arrow} {direction.upper()} SETUP: {sym}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🔻 Entry Zone : {entry}\n"
            f"⛔️ Stop Loss  : {sl}\n"
            f"\n"
            f"🎯 Take Profit Targets:\n"
            f"• TP1: {tp1} (Safe)\n"
            f"• TP2: {tp2} (Mid)\n"
            f"• TP3: {tp3} (Max)\n"
            f"\n"
            f"📐 Trade Info:\n"
            f"Leverage: {leverage}x\n"
            f"Win Prob: {win_prob:.0f}%\n"
            f"Market:   {trend_label}\n"
            f"\n"
            f"💡 Signal Catalyst:\n"
            f"{catalyst_text}\n"
            f"\n"
            f"{self.CHANNEL}"
        )

    def tp_hit(
        self,
        raw_sym: str,
        tp_label: str,
        price: float,
        profit_pct: float,
    ) -> str:
        sym = self._clean_symbol(raw_sym)
        return (
            f"✅ TAKE PROFIT HIT: {sym}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Target: {tp_label} 🎯\n"
            f"Price: {price}\n"
            f"Profit: +{profit_pct:.1f}%\n"
            f"\n"
            f"{self.CHANNEL}"
        )

    def sl_hit(
        self,
        raw_sym: str,
        price: float,
        loss_pct: float,
    ) -> str:
        sym = self._clean_symbol(raw_sym)
        return (
            f"🛡 STOP LOSS HIT: {sym}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Price: {price}\n"
            f"Loss: -{loss_pct:.1f}%\n"
            f"Risk managed successfully.\n"
            f"\n"
            f"{self.CHANNEL}"
        )
