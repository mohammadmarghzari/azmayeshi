# Portfolio360 (پورتفولیو ۳۶۰)

اپلیکیشن شخصی تحلیل پرتفوی، اختیار معامله و ابزارهای بازار ایران — نسخه بازنویسی‌شده با **Kotlin Multiplatform + Compose Multiplatform**، برای **دسکتاپ (ویندوز/مک/لینوکس)** و **اندروید**، از روی نسخه اولیه‌ی Python/Streamlit همین پروژه (`app.py`).

> این یک اپ کاملاً شخصی است و برای ارائه‌ی عمومی در نظر گرفته نشده.

## امکانات

همه‌ی ۱۶ بخش نسخه‌ی پایتون، با UI حرفه‌ای Compose و نمودارهای اختصاصی (Canvas-based) پیاده‌سازی شده‌اند:

- **تخصیص پرتفوی** — بهینه‌سازی با ۶ روش (بیشترین شارپ، کمترین واریانس، مونت‌کارلو/CVaR، وزن برابر، ریسک پاریتی، بشکه‌ای طالب)، Pie و Treemap
- **ریسک و بازده** — Sharpe، Calmar، CVaR، Max Drawdown، Underwater chart، هج با Protective Put
- **نمودار قیمت** و ماتریس همبستگی
- **مقایسه سبک‌ها** — مقایسه هر ۶ روش تخصیص در کنار هم
- **Efficient Frontier** — شبیه‌سازی ۱۵۰۰ پرتفوی تصادفی + مرز کارایی + Rolling Sharpe
- **اختیار پیشرفته** — Covered Call، Protective Put، Iron Condor، Rolling Covered Call
- **Black-Litterman** — دیدگاه‌های شخصی + Factor Exposure
- **Stress Test & Monte Carlo** — بازپخش ۶ بحران تاریخی + شبیه‌سازی ۴۰۰ مسیر آینده
- **ری‌بالانس** و تشخیص رژیم همبستگی بازار
- **Benchmark** — مقایسه با SPY/QQQ/BTC/... (آلفا، بتا، Tracking Error)
- **داده زنده** — شاخص Fear & Greed، اخبار Yahoo Finance، Seasonality
- **ذخیره پرتفوی**، **هشدار قیمت/Fear&Greed**
- **ابزار ایران** — نرخ واقعی دلار (تورم/طلا)، حباب گواهی سپرده کالایی، P&L هدف قیمتی
- **اختیار بورس کالا** — قیمت‌گذاری Black-Scholes، IV Solver، ۶ استراتژی ترکیبی، Option Chain، چک‌لیست معامله
- **IME Live** — تابلوی زنده گواهی سپرده کالایی، دفتر سفارشات، نمودار شمعی، رادار مقایسه، تحلیل‌گر سیگنال

## معماری پروژه

```
core/          ماژول Kotlin/JVM خالص — تمام فرمول‌های مالی + شبکه (بدون UI)
  math/        بهینه‌سازی پرتفوی، Black-Scholes، Black-Litterman، Monte Carlo، ...
  network/     کلاینت‌های Ktor برای Yahoo Finance، CNN Fear&Greed، RSS اخبار، IME API
  model/       مدل‌های داده مشترک
composeApp/    اپ Kotlin Multiplatform (Compose) — کد UI مشترک بین دسکتاپ و اندروید
  commonMain/  تم، نمودارهای اختصاصی (Canvas)، ۱۶ صفحه، ناوبری تطبیقی
  androidMain/ نقطه‌ورود اندروید (MainActivity)
  desktopMain/ نقطه‌ورود دسکتاپ (Compose Desktop window)
```

فایل‌های نسخه‌ی اصلی پایتون (`app.py`, `options/`, `assets/`, `requirements.txt`) دست‌نخورده باقی مانده‌اند تا به‌عنوان مرجع در دسترس باشند؛ توسعه‌ی آینده روی نسخه‌ی Kotlin ادامه پیدا می‌کند.

## اجرا روی دسکتاپ

```bash
./gradlew :composeApp:run
```

برای ساخت نصب‌کننده‌ی بومی (dmg/msi/deb):

```bash
./gradlew :composeApp:createDistributable
```

## اجرا روی اندروید

پروژه را در **Android Studio** باز کنید و روی یک دستگاه/شبیه‌ساز اجرا کنید، یا از خط فرمان:

```bash
./gradlew :composeApp:assembleDebug
```

فایل APK در `composeApp/build/outputs/apk/debug/` ساخته می‌شود.

> ساخت اندروید نیازمند دسترسی شبکه به Google Maven است (`google()`)؛ در محیط‌های محدود (مثل سندباکس این توسعه) این مرحله در GitHub Actions به‌صورت خودکار تست می‌شود (`.github/workflows/build.yml`).

## تست ماژول اصلی

```bash
./gradlew :core:test
```

## پشته فناوری

Kotlin 2.0 · Compose Multiplatform 1.7 · Ktor Client (CIO) · kotlinx.coroutines / serialization / datetime

---
ساخته شده با ❤️ توسط محمد مرغزاری — بازنویسی‌شده با Claude


## مستندات توسعه (برای ابزارهای AI و توسعه‌دهنده‌ها)

- **اسکیل اصلی پروژه**: [`.claude/skills/portfolio360-master/`](.claude/skills/portfolio360-master/SKILL.md) — استانداردهای معماری، UI، مالی و خط تولید تحویل. این اسکیل به‌صورت پیش‌فرض روی همه کارهای این مخزن اعمال می‌شود.
- [`CLAUDE.md`](CLAUDE.md) — نقطه ورود دستیارهای کدنویسی؛ به اسکیل اصلی ارجاع می‌دهد.
- چک‌لیست پیش از هر کامیت: [`checklist.md`](.claude/skills/portfolio360-master/checklist.md)
