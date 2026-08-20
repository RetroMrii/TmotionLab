Baseline: baseline_tfidf_dair_v001

Test accuracy: 0.8320
Test macro F1: 0.7301
Test weighted F1: 0.8208

Primary weaknesses:
- surprise recall: 0.2576
- love recall: 0.4969
- joy acts as the dominant confusion target

Largest confusion pairs:
- love -> joy: 66
- anger -> joy: 36
- sadness -> joy: 35
- anger -> sadness: 31
- fear -> joy: 29
- fear -> sadness: 28
- surprise -> joy: 26
- surprise -> fear: 15