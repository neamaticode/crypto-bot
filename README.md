# 💎 VIP Crypto Signal Bot v2.0

A professional-grade, AI-powered cryptocurrency futures signal bot for Binance.  
Delivers ultra-precise trading signals via Telegram with Smart Money Concepts,  
multi-timeframe analysis, ensemble machine learning, and advanced risk management.

---

## 🇬🇧 English Setup Guide

### Requirements
- Python 3.10 or newer
- A Binance Futures account with API access
- A Telegram bot token and channel/group chat ID

### 1 – Clone & install dependencies
```bash
git clone https://github.com/neamaticode/crypto-bot.git
cd crypto-bot
pip install -r requirements.txt
```

### 2 – Configure the bot
Open `vippp.py` and fill in the following values near the top of the file:

| Variable | Description |
|---|---|
| `TELEGRAM_TOKEN` | Your Telegram bot token (from @BotFather) |
| `TELEGRAM_CHAT_ID` | Your channel or group chat ID |
| `API_CONFIG['apiKey']` | Binance API key (Futures read permission required) |
| `API_CONFIG['secret']` | Binance API secret |
| `INITIAL_ACCOUNT_BALANCE` | Your starting account size in USDT |

### 3 – Run the bot
```bash
python vippp.py
```

To keep it running in the background on a Linux server:
```bash
nohup python vippp.py > bot.log 2>&1 &
# or with screen:
screen -S cryptobot
python vippp.py
# Ctrl+A then D to detach
```

### Features at a glance
- ✅ Scans **all** USDT perpetual futures pairs on Binance
- 📊 **16+ technical indicators** – RSI, MACD, Bollinger Bands, EMAs (9/21/50/100/200), ATR, ADX, Stochastic RSI, OBV, Volume Ratio, VWAP, Ichimoku Cloud, Supertrend, Fibonacci, Support/Resistance
- 🧠 **Smart Money Concepts** – Order Blocks, Fair Value Gaps, Liquidity Zones, BOS/CHoCH, RSI Divergence
- ⏱ **Multi-timeframe analysis** – 15m, 1h, 4h, 1d confirmation
- 🤖 **AI ensemble model** – GradientBoosting + RandomForest (auto-retrains daily)
- 🛡 **Advanced risk management** – 1% risk per trade, trailing stop, breakeven, partial TPs
- 📩 **Professional Telegram messages** with confidence bars, full reasoning, and trade updates
- �� **Daily / Weekly / Monthly reports** at 23:00 Afghanistan time (UTC+4:30)
- 🗄 **SQLite database** for full trade history and AI training data

---

## 🇦🇫 راهنمای نصب (دری / فارسی)

### پیش‌نیازها
- Python نسخه 3.10 یا بالاتر
- حساب Binance Futures با دسترسی API
- توکن ربات تلگرام و آیدی کانال یا گروه

### ۱ – دریافت کد و نصب وابستگی‌ها
```bash
git clone https://github.com/neamaticode/crypto-bot.git
cd crypto-bot
pip install -r requirements.txt
```

### ۲ – تنظیم ربات
فایل `vippp.py` را باز کنید و مقادیر زیر را در بالای فایل پر کنید:

| متغیر | توضیح |
|---|---|
| `TELEGRAM_TOKEN` | توکن ربات تلگرام شما (از @BotFather) |
| `TELEGRAM_CHAT_ID` | آیدی کانال یا گروه تلگرام |
| `API_CONFIG['apiKey']` | کلید API بایننس (نیاز به دسترسی Futures) |
| `API_CONFIG['secret']` | رمز مخفی API بایننس |
| `INITIAL_ACCOUNT_BALANCE` | موجودی اولیه حساب به USDT |

### ۳ – اجرای ربات
```bash
python vippp.py
```

برای اجرای دائمی روی سرور لینوکس:
```bash
nohup python vippp.py > bot.log 2>&1 &
# یا با screen:
screen -S cryptobot
python vippp.py
# کلیدهای Ctrl+A سپس D برای جدا شدن
```

### قابلیت‌های اصلی
- ✅ اسکن **تمام** جفت‌ارزهای USDT فیوچرز بایننس
- 📊 **۱۶+ اندیکاتور تکنیکال** – RSI، MACD، بولینگر بندز، EMAها، ATR، ADX، Stochastic RSI، OBV، نسبت حجم، VWAP، ابر ایچیموکو، سوپرترند، فیبوناچی، حمایت/مقاومت
- 🧠 **مفاهیم پول هوشمند** – Order Block، Fair Value Gap، مناطق نقدینگی، BOS/CHoCH، واگرایی RSI
- ⏱ **تحلیل چند تایم‌فریم** – تأیید در ۱۵ دقیقه، ۱ ساعت، ۴ ساعت، روزانه
- 🤖 **مدل هوش مصنوعی** – GradientBoosting + RandomForest (بازآموزی روزانه خودکار)
- 🛡 **مدیریت ریسک پیشرفته** – ۱٪ ریسک در هر ترید، تریلینگ استاپ، بریک‌ایون، برداشت جزئی
- 📩 **پیام‌های حرفه‌ای تلگرام** با نوار اطمینان، دلایل کامل و به‌روزرسانی معاملات
- 📅 **گزارش روزانه / هفتگی / ماهانه** ساعت ۲۳:۰۰ وقت افغانستان (UTC+4:30)
- 🗄 **پایگاه داده SQLite** برای تاریخچه کامل معاملات و داده‌های آموزش AI

---

## ⚠️ Disclaimer / سلب مسئولیت

> **EN:** This bot is for educational purposes only. Trading cryptocurrencies involves significant financial risk. Past performance does not guarantee future results. Always manage your risk and never invest more than you can afford to lose.

> **دری:** این ربات صرفاً برای مقاصد آموزشی است. معامله ارزهای دیجیتال دارای ریسک مالی بالاست. عملکرد گذشته تضمینی برای آینده نیست. همیشه ریسک خود را مدیریت کنید.
