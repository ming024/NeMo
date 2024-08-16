# Dataset Preparation for Speech-to-Speech Translation

This README will walk you through the process of creating Es-En, Fr-En and De-En S2ST datasets in the shar format.

## Requirements
- Wav files of the source utterances (in Es, Fr, De, respectively) which can be found at ``10.110.40.244:/mnt/drive1/data/s2s/{es,fr,de}/source_wav/``
- Synthetic wav files of the target utterances (in English) which can be found at ``10.110.40.244:/mnt/drive1/data/s2s/{es,fr,de}/target_wav/``. Thanks @subhankarg for help me create the synthetic data with the NeMo TTS model!
- The manifest file used for synthetic data generation ``10.110.40.244:/mnt/drive1/data/s2s/{es,fr,de}/manifest_{es,fr,de}_to_rodney_no_emotion_context_Rodney_44khz_WIZWIKI_RODNEY_WIZWIKI_005619_s2s.json/``

## Creating target codecs
The first step is converting the target wav files into codecs, which will be used as the target labels when trainig the LM.
We use the  ``SpeechCodec_2402.nemo`` nemo audio codec model with 8 codebooks.
To generate the codecs, in the Docker environment, run
```
python scripts/wav_to_codec.py --manifest_paths /workspace/data/s2s/{es,fr,de}/manifest_{es,fr,de}_to_rodney_no_emotion_context_Rodney_44khz_WIZWIKI_RODNEY_WIZWIKI_005619_s2s.json --base_data_dir /workspace/data/s2s/{es,fr,de}/target_wav/ --out_dir /workspace/data/s2s_codec/{es,fr,de}/ --codec_model_path PATH_TO_CODEC_MODEL
```

The generated codec files will be saved to ``/workspace/data/s2s_codec/{es,fr,de}/``.

## Modifying manifest files
Then, we have to add the codec paths into the manifest files. Run
```
python scripts/add_codec_path.py --input_manifest /workspace/data/s2s/{es,fr,de}/manifest_{es,fr,de}_to_rodney_no_emotion_context_Rodney_44khz_WIZWIKI_RODNEY_WIZWIKI_005619_s2s.json --output_manifest /workspace/data/s2s/{es,fr,de}/manifest_{es,fr,de}_to_rodney_no_emotion_context_Rodney_44khz_WIZWIKI_RODNEY_WIZWIKI_005619_s2s_codec.json --codec_base_dir /workspace/data/s2s_codec/{es,fr,de}/codecs/nemo_codec_bw_6.0/
```

The modified manifest file will be saved at ``/workspace/data/s2s/{es,fr,de}/manifest_{es,fr,de}_to_rodney_no_emotion_context_Rodney_44khz_WIZWIKI_RODNEY_WIZWIKI_005619_s2s_codec.json``.

## Running ASR forced-alignment to filter out low-quality synthetic data
Sometimes, synthetic data does not follow the transcription perfectly.
To filter out low-quality synthetic data, we use [wav2vec2 forced alignment tool](https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html) to score the alignment between the text transcript and synthetic speech.
Run
```
python scripts/asr_filtering.py --input_manifest /workspace/data/s2s/{es,fr,de}/manifest_{es,fr,de}_to_rodney_no_emotion_context_Rodney_44khz_WIZWIKI_RODNEY_WIZWIKI_005619_s2s_codec.json --output_manifest /workspace/data/s2s/{es,fr,de}/manifest_{es,fr,de}_to_rodney_no_emotion_context_Rodney_44khz_WIZWIKI_RODNEY_WIZWIKI_005619_s2s_codec_forced_ali_score.json --audio_folder /workspace/data/s2s/{es,fr,de}/target_wav/
```
The scores are then added to the manifest files saved at ``/workspace/data/s2s/{es,fr,de}/manifest_{es,fr,de}_to_rodney_no_emotion_context_Rodney_44khz_WIZWIKI_RODNEY_WIZWIKI_005619_s2s_codec_forced_ali_score.json``.
When training of the S2ST model, the scores can be used as a criteria of data selection to ensure high training data quality.

## Creating shars
The last step is packing up the dataset into shars.
Simply run
```
python script/create_shar.py --manifest /workspace/data/s2s/{es,fr,de}/manifest_{es,fr,de}_to_rodney_no_emotion_context_Rodney_44khz_WIZWIKI_RODNEY_WIZWIKI_005619_s2s_codec_forced_ali_score.json --source_wav_dir /workspace/data/s2s/{es,fr,de}/source_wav/ --out_shar_dir /workspace/data/s2s_shars/{es,fr,de}/ --codec_dir /workspace/data/s2s_codec/{es,fr,de}/codecs/nemo_codec_bw_6.0/ --codec_np_dir /workspace/data/s2s_codec/{es,fr,de}/codecs/numpy/ --source_lang {es,fr,de}
```

The created shars will be saved at
``/workspace/data/s2s_shars/{es,fr,de}/``.