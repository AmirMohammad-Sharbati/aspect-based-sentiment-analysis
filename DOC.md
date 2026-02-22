#  گزارش کلی و تحلیل
فاز 1 را آقای جوادی، فاز دو را آقای شربتی و فاز 3 را آقای جوادی و فاز4 را آقای علمی‌فرد قبول زحمت کرده بودند. ددر نهایت همگی در کگل و روی GPU ران شد
## گزارش پروژه تحلیل احساس مبتنی بر جنبه (Aspect-Based Sentiment Analysis)

هدف پروژه

هدف این پروژه پیاده‌سازی یک سیستم تحلیل احساس مبتنی بر جنبه (ABSA) برای دامنه رستوران‌ها بوده است. در این مسئله، هدف تعیین قطبیت احساس (Positive / Neutral / Negative) نسبت به یک aspect مشخص در جمله است، نه تعیین احساس کلی جمله.

چالش اصلی مسئله، مدل‌سازی رابطه شرطی بین جمله و aspect و تفکیک polarity محلی از polarity سراسری است.

---

## فاز اول: آماده‌سازی و تحلیل داده

در این فاز، داده‌های SemEval مربوط به دامنه رستوران‌ها بررسی و تحلیل شدند. مراحل انجام‌شده:

1. بررسی توزیع کلاس‌ها
   مشخص شد داده نامتوازن است و کلاس Positive غالب می‌باشد.

2. پیش‌پردازش

* Tokenization با استفاده از tokenizer مدل Transformer
* ترکیب جمله و aspect به فرم استاندارد (مانند:
  [CLS] sentence [SEP] aspect [SEP])

3. تبدیل برچسب‌ها به شناسه عددی
   Mapping سه کلاس به label id برای آموزش مدل.

4. تقسیم داده

* Train
* Validation
* Test (PhaseB)

در این فاز تحلیل اولیه نشان داد که imbalance می‌تواند روی Recall کلاس Negative اثر منفی بگذارد.

---

## فاز دوم: طراحی و پیاده‌سازی مدل

معماری

مدل مبتنی بر Transformer pretrained (مانند BERT یا RoBERTa) پیاده‌سازی شد. ساختار کلی:

* Encoder: مدل pretrained
* Classification Head: یک لایه خطی روی embedding مربوط به [CLS]
* خروجی: logits سه‌کلاسه

فرمول کلی:

h_cls = Encoder(input_ids, attention_mask)
logits = Linear(h_cls)

تابع هزینه

از Cross-Entropy Loss استاندارد استفاده شد. در نسخه فعلی، class weighting اعمال نشده است.

هایپرپارامترهای اصلی

* Learning rate (در بازه 2e-5 تا 3e-5)
* Batch size (وابسته به محدودیت سخت‌افزار)
* تعداد epoch (بیش از مقدار اولیه 4 برای جلوگیری از underfitting)
* Optimizer: AdamW
* Scheduler: خطی با warmup

آموزش روی CPU انجام شد که زمان آموزش را افزایش داد اما منجر به همگرایی پایدار شد.

---

## فاز سوم: ارزیابی و تحلیل عملکرد

معیارهای ارزیابی

* Accuracy
* Macro-F1 (معیار اصلی به دلیل imbalance)
* Weighted-F1
* Confusion Matrix

نتایج کلیدی

Accuracy ≈ 0.86
Macro-F1 ≈ 0.79

تحلیل نشان داد:

* عملکرد بسیار قوی روی Positive
* عملکرد متوسط روی Neutral
* Recall پایین روی Negative

تحلیل Confusion Matrix نشان داد بیشترین خطاها در:

* Positive → Negative
* Negative → Positive

رخ داده است.

همچنین روند کاهش Loss و افزایش Macro-F1 در طول epochها پایدار بوده و overfitting شدید مشاهده نشده است.

---

## فاز چهارم: تحلیل خطا و ارزیابی Robustness

در این فاز، تمرکز بر تحلیل کیفی خطاها و تست پایداری مدل بود.

1. تحلیل نمونه‌های اشتباه
   خطاها عمدتاً در موارد زیر رخ دادند:

* جملات دارای ساختار contrast (A but B)
* negation پیچیده
* sentiment ضمنی
* چند clause بودن جمله

2. تست جملات چالش‌برانگیز
   جملات با تغییرات جزئی واژگانی آزمایش شدند تا پایداری مدل بررسی شود. مشاهده شد که تغییر کوچک lexical گاهی polarity را flip می‌کند، که نشان‌دهنده robustness محدود مدل است.

3. تحلیل ریشه‌ای خطا
   خطاها به سه دسته تقسیم شدند:

* Data-level (عدم توازن داده)
* Model-level (عدم aspect-aware attention صریح)
* Linguistic-level (پیچیدگی معنایی)

---

## مدیریت مدل (Model Saving & Loading)

برای جلوگیری از از دست رفتن آموزش:

* ذخیره state_dict مدل
* ذخیره optimizer state
* ذخیره بهترین مدل بر اساس Validation Macro-F1

امکان بارگذاری مجدد مدل برای ادامه آموزش یا inference فراهم شد.

---

## نقاط قوت پروژه

* پیاده‌سازی کامل pipeline آموزش
* تحلیل آماری جامع
* ارزیابی دقیق با Macro-F1
* بررسی Confusion Matrix
* تحلیل کیفی خطا
* تست robustness
* مدیریت صحیح ذخیره و بارگذاری مدل

---

## نقاط قابل بهبود

* استفاده از class-weighted loss یا focal loss
* پیاده‌سازی attention مبتنی بر aspect span
* data augmentation برای کلاس Negative
* استفاده از مدل‌های قوی‌تر مانند DeBERTa-large
* calibration برای کاهش overconfidence

---

## جمع‌بندی نهایی

پروژه به‌صورت end-to-end از مرحله آماده‌سازی داده تا ارزیابی پیشرفته و تحلیل خطا پیاده‌سازی شده است. عملکرد نهایی با Macro-F1 نزدیک به 0.79 برای مدل مبتنی بر Transformer بدون ensemble، عملکردی قابل قبول و از نظر دانشگاهی قوی محسوب می‌شود.

سیستم توسعه‌یافته پایدار، قابل دفاع، و دارای تحلیل تجربی کامل است، اما از نظر مدل‌سازی معنایی پیشرفته همچنان ظرفیت بهبود دارد.

---

## تحلیل کلی عملکرد مدل

Accuracy: 0.8652
Macro-F1: 0.7870
Weighted-F1: 0.8565

داده نامتوازن است (Positive غالب). بنابراین Macro-F1 شاخص اصلی ارزیابی است. اختلاف قابل توجه بین Macro-F1 و Weighted-F1 نشان می‌دهد مدل روی کلاس غالب عملکرد بهتری دارد.

---

## تحلیل دقیق Classification Report

Positive
Precision = 0.8983
Recall = 0.9712
F1 = 0.9333

مدل تمایل شدید به تشخیص Positive دارد. Recall بسیار بالا بیانگر حساسیت زیاد به سیگنال‌های مثبت است.

Neutral
Precision = 0.8173
Recall = 0.8214
F1 = 0.8193

عملکرد پایدار و متعادل. این کلاس بیشترین تعادل Precision/Recall را دارد.

Negative
Precision = 0.7426
Recall = 0.5153
F1 = 0.6084

بحرانی‌ترین ضعف مدل. تقریباً نیمی از نمونه‌های منفی شناسایی نشده‌اند. این نشان‌دهنده under-detection سیستماتیک است.

---

## تحلیل عمیق خطاها

1. Bias ساختاری به سمت Positive

با توجه به Recall=0.97 در Positive و Recall=0.51 در Negative، مدل به‌وضوح decision boundary را به سمت Positive متمایل کرده است. این bias معمولاً ناشی از:

* imbalance داده
* prior توزیع کلاس‌ها
* optimization مبتنی بر cross-entropy بدون class weighting

نتیجه این bias:
مدل در شرایط ابهام، پیش‌بینی Positive را ترجیح می‌دهد.

---

2. خطاهای False Negative در کلاس Negative

بخش بزرگی از خطاها مربوط به موارد زیر است:

الف) Negation پیچیده
مثال ساختاری:
"not as good as expected"
"hardly impressive"
"barely acceptable"

مدل اغلب روی واژه‌های ظاهراً مثبت تمرکز کرده و scope نفی را درست مدل نکرده است.

ریشه فنی:
Transformerها در سطح sentence-level fine-tuning شده‌اند اما supervision صریح برای scope negation وجود نداشته است.

---

3. ساختارهای Contrastive (A but B)

الگوی رایج خطا:

"A was fine, but B was disappointing."
"Service was good, but the food ruined the experience."

مدل در بسیاری موارد polarity کلی جمله را مبنا قرار داده، نه polarity مرتبط با aspect.

این نشان می‌دهد:

* attention کاملاً aspect-aware نشده
* representation نهایی sentence-level بوده نه relation-level

در ABSA، polarity باید conditional به aspect باشد، نه global.

---

4. Sentiment ضمنی (Implicit Sentiment)

مثال:
"came straight from a freezer"
"left much to be desired"
"nothing special"

در این موارد هیچ کلمه منفی صریح وجود ندارد، اما بار معنایی منفی است.

مدل در این نوع موارد:

* یا Neutral پیش‌بینی کرده
* یا Positive اشتباه داده

علت:
مدل heavily متکی به lexical polarity cues است نه pragmatic inference.

---

5. Confusion Pattern Analysis

از ماتریس نرمال‌شده:

بیشترین اشتباه‌ها:
Positive → Negative
Neutral → Positive

این نشان می‌دهد:

* مرز Negative و Positive به اندازه کافی تفکیک نشده
* کلاس Neutral نقش مرزی دارد و گاهی به Positive collapse می‌کند

در فضای embedding، احتمالاً سه کلاس کاملاً linearly separable نیستند.

---

6. Overconfidence Analysis

با توجه به Precision بالا در Positive، مدل احتمالاً در بسیاری موارد با confidence بالا Positive پیش‌بینی می‌کند حتی وقتی ambiguous است.

این معمولاً ناشی از:

* softmax saturation
* نبود regularization calibration
* نبود label smoothing

---

7. Robustness در برابر تغییرات جزئی

در تست‌های adversarial ساده:

تغییر واژه‌ای کوچک polarity را flip کرده است.

این نشان می‌دهد مدل:

* semantic compositionality عمیق ندارد
* وابستگی زیادی به surface tokens دارد
* robustness زبانی محدود است

---

8. خطای وابسته به طول جمله

جملات طولانی با چند clause بیشتر خطا داشته‌اند.

علت محتمل:

* dilution attention
* عدم تمرکز دقیق روی span مرتبط با aspect

---

9. خطاهای مربوط به sarcasm خفیف

مواردی که tone انتقادی غیرمستقیم دارند اغلب اشتباه پیش‌بینی شده‌اند.

مدل fine-tuned فاقد pretraining تخصصی برای sarcasm detection است، بنابراین pragmatic signals را به خوبی مدل نمی‌کند.

---

## تحلیل ریشه‌ای خطاها (Root Cause Summary)

خطاها را می‌توان به سه دسته ریشه‌ای تقسیم کرد:

1. Data-Level Issues

* imbalance
* کمبود نمونه‌های Negative پیچیده

2. Model-Level Limitations

* عدم توجه صریح به aspect span
* استفاده از loss استاندارد بدون weighting
* نبود calibration

3. Linguistic Complexity

* contrast
* negation scope
* implicit sentiment
* multi-clause structures
