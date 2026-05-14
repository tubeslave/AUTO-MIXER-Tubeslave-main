# Model Card: MuQ-Eval

## Model Details

- **Model name:** MuQ-Eval (A1: Frozen MuQ + MSE)
- **Model type:** Per-sample audio quality predictor
- **Architecture:** MuQ-310M (frozen) + attention pooling + 2-layer MLP
- **Encoder:** OpenMuQ/MuQ-large-msd-iter (Wav2Vec2-Conformer, 310M params)
- **Trainable parameters:** ~1M (attention pooling + prediction heads)
- **Input:** Audio waveform at 24 kHz, 10-second clips
- **Output:** Quality scores on 1-5 MOS scale (MI: Musical Impression, TA: Textual Alignment)
- **License:** MIT

## Intended Use

- Automatic quality assessment of generated music audio
- Ranking and comparing text-to-music generation systems
- Quality filtering in music generation pipelines
- Research on music evaluation metrics

## Performance

Evaluated on MusicEval (2,748 clips, 31 TTM systems, 5-fold CV):

| Level | Metric | Value | 95% CI |
|-------|--------|-------|--------|
| System (31 models) | SRCC | 0.957 | [0.898, 0.986] |
| System (31 models) | PCC | 0.960 | [0.916, 0.980] |
| Utterance (~385/fold) | SRCC (MI) | 0.838 | [0.807, 0.865] |
| Utterance (~385/fold) | PCC (MI) | 0.828 | [0.796, 0.854] |

Reference comparison: Audiobox Aesthetics achieves r = 0.200 on music generation preferences.

## Training Data

- **Primary:** MusicEval (BAAI/MusicEval) -- 2,748 clips from 31 TTM systems, 13,740 expert MOS ratings
- **Annotations:** Musical Impression (MI) and Textual Alignment (TA) on 1-5 Likert scale
- **CV scheme:** 5-fold stratified by TTM model

## Limitations

- **Single dataset:** Trained and evaluated only on MusicEval. Cross-dataset generalization is untested.
- **Expert ratings only:** Predicts expert MOS, which may not align with general listener preferences.
- **Audio-only:** Cannot assess text-audio alignment (TA scores are lower, SRCC ~0.59) because the model processes only audio, not text prompts.
- **Degradation sensitivity:** Not tested for monotonic response to controlled audio degradations.
- **Quality range:** MusicEval spans a wide quality range (poor to near-human). Performance on fine-grained quality discrimination (e.g., between two high-quality systems) is unknown.
- **Language/culture:** Training data is primarily Western music from English-language TTM systems.

## Ethical Considerations

- This metric reflects the biases present in the MusicEval expert ratings panel.
- Should not be used as the sole criterion for music quality; human evaluation remains essential.
- Scores are specific to generated music quality and should not be applied to judge artistic merit of human-created music.

## Computational Requirements

- **Inference:** ~35 ms per 10-second clip on GPU, ~3 GB VRAM
- **Training:** ~2 GPU-hours on RTX 4080 16 GB for 5-fold CV
- **Mixed precision:** bf16 supported
