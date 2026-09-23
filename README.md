# RetinaScan

AI-assisted diabetic retinopathy screening from retina photographs.

## Why this project

Diabetes affects an estimated 1 in 4 adults in Pakistan — one of the highest rates in the world. One of its most serious complications is diabetic retinopathy: damage to the retina's blood vessels that can lead to permanent blindness if caught too late.

Regular eye screening catches it early, but ophthalmologists are concentrated in major cities, and many people with diabetes never get screened until symptoms are already advanced. This project is an AI screening tool aimed at helping close that gap.

## How it works
xccccccccccccccccccccccccccccvnxkvnasdlkvdkfvldakfmvldf
Upload a retina photograph and get an instant assessment across five severity levels (No DR → Mild → Moderate → Severe → Proliferative), along with a full confidence breakdown rather than just a single label.

## The real engineering challenge

Training a model wasn't the hard part — handling a deeply imbalanced dataset responsibly was.

Nearly half the training images showed no disease, while the most dangerous cases (Severe, Proliferative) made up less than 15% combined. A naive model trades safety for a good-looking accuracy score: high overall accuracy, but it quietly misses most of the severe cases — exactly backwards for a screening tool.

So the model was optimized for **macro recall** instead of raw accuracy, explicitly rewarding it for catching rare, dangerous cases rather than just the easy majority class.

**Result:**
- Recall on Severe cases improved from 44% → 69%
- Recall on Proliferative cases improved from 34% → 59%
- Only a modest trade-off in overall accuracy

## Tech stack

**Model:** PyTorch + EfficientNet-B0 (transfer learning)
**Backend:** FastAPI
**Frontend:** Next.js, TypeScript

## Important note

This is a screening aid built for demonstration, not a diagnostic tool — and the app states this explicitly. It's a real example of what responsible medical AI design looks like: understanding that not all errors are equal, and building for the failure mode that actually matters.

## Live demo

[retinascan-three.vercel.app](https://retinascan-three.vercel.app)

## Status

Deployed and live
