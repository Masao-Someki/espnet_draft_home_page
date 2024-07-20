import{_ as s}from"./plugin-vue_export-helper-DlAUqK2U.js";import{c as n,o as e,a}from"./app-KOUU_Wij.js";const i={},l=a(`<h1 id="cmu-11492-11692-spring-2023-speech-enhancement" tabindex="-1"><a class="header-anchor" href="#cmu-11492-11692-spring-2023-speech-enhancement"><span>CMU 11492/11692 Spring 2023: Speech Enhancement</span></a></h1><p>In this demonstration, we will show you some demonstrations of speech enhancement systems in ESPnet.</p><p>Main references:</p><ul><li><a href="https://github.com/espnet/espnet" target="_blank" rel="noopener noreferrer">ESPnet repository</a></li><li><a href="https://espnet.github.io/espnet/" target="_blank" rel="noopener noreferrer">ESPnet documentation</a></li><li><a href="https://github.com/espnet/espnet/tree/master/egs2/TEMPLATE/enh1" target="_blank" rel="noopener noreferrer">ESPnet-SE repo</a></li></ul><p>Author:</p><ul><li>Siddhant Arora (siddhana@andrew.cmu.edu)</li></ul><p>The notebook is adapted from this <a href="https://colab.research.google.com/drive/1faFfqWNFe1QW3Q1PMwRXlNDwaBms__Ho?usp=sharing" target="_blank" rel="noopener noreferrer">Colab</a></p><h2 id="❗important-notes❗" tabindex="-1"><a class="header-anchor" href="#❗important-notes❗"><span>❗Important Notes❗</span></a></h2><ul><li>We are using Colab to show the demo. However, Colab has some constraints on the total GPU runtime. If you use too much GPU time, you may not be able to use GPU for some time.</li><li>There are multiple in-class checkpoints ✅ throughout this tutorial. <strong>Your participation points are based on these tasks.</strong> Please try your best to follow all the steps! If you encounter issues, please notify the TAs as soon as possible so that we can make an adjustment for you.</li><li>Please submit PDF files of your completed notebooks to Gradescope. You can print the notebook using <code>File -&gt; Print</code> in the menu bar.You also need to submit the spectrogram and waveform of noisy and enhanced audio files to Gradescope.</li></ul><h1 id="contents" tabindex="-1"><a class="header-anchor" href="#contents"><span>Contents</span></a></h1><p>Tutorials on the Basic Usage</p><ol><li><p>Install</p></li><li><p>Speech Enhancement with Pretrained Models</p></li></ol><blockquote><p>We support various interfaces, e.g. Python API, HuggingFace API, portable speech enhancement scripts for other tasks, etc.</p></blockquote><p>2.1 Single-channel Enhancement (CHiME-4)</p><p>2.2 Enhance Your Own Recordings</p><p>2.3 Multi-channel Enhancement (CHiME-4)</p><ol start="3"><li>Speech Separation with Pretrained Models</li></ol><p>3.1 Model Selection</p><p>3.2 Separate Speech Mixture</p><ol start="4"><li>Evaluate Separated Speech with the Pretrained ASR Model</li></ol><p>Tutorials on the Basic Usage</p><h2 id="install" tabindex="-1"><a class="header-anchor" href="#install"><span>Install</span></a></h2><p>Different from previous assignment where we install the full version of ESPnet, we use a lightweight ESPnet package, which mainly designed for inference purpose. The installation with the light version can be much faster than a full installation.</p><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span>import locale</span></span>
<span class="line"><span>locale.getpreferredencoding = lambda: &quot;UTF-8&quot;</span></span>
<span class="line"><span>%pip uninstall torch</span></span>
<span class="line"><span>%pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117</span></span>
<span class="line"><span>%pip install -q git+https://github.com/espnet/espnet</span></span>
<span class="line"><span>%pip install -q espnet_model_zoo</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h2 id="speech-enhancement-with-pretrained-models" tabindex="-1"><a class="header-anchor" href="#speech-enhancement-with-pretrained-models"><span>Speech Enhancement with Pretrained Models</span></a></h2><h3 id="single-channel-enhancement-the-chime-example" tabindex="-1"><a class="header-anchor" href="#single-channel-enhancement-the-chime-example"><span>Single-Channel Enhancement, the CHiME example</span></a></h3><h3 id="task1-✅-checkpoint-1-1-point" tabindex="-1"><a class="header-anchor" href="#task1-✅-checkpoint-1-1-point"><span>Task1 (✅ Checkpoint 1 (1 point))</span></a></h3><p>Run inference of pretrained single-channel enhancement model.</p><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span># Download one utterance from real noisy speech of CHiME4</span></span>
<span class="line"><span>!gdown --id 1SmrN5NFSg6JuQSs2sfy3ehD8OIcqK6wS -O /content/M05_440C0213_PED_REAL.wav</span></span>
<span class="line"><span>import os</span></span>
<span class="line"><span></span></span>
<span class="line"><span>import soundfile</span></span>
<span class="line"><span>from IPython.display import display, Audio</span></span>
<span class="line"><span>mixwav_mc, sr = soundfile.read(&quot;/content/M05_440C0213_PED_REAL.wav&quot;)</span></span>
<span class="line"><span># mixwav.shape: num_samples, num_channels</span></span>
<span class="line"><span>mixwav_sc = mixwav_mc[:,4]</span></span>
<span class="line"><span>display(Audio(mixwav_mc.T, rate=sr))</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h4 id="download-and-load-the-pretrained-conv-tasnet" tabindex="-1"><a class="header-anchor" href="#download-and-load-the-pretrained-conv-tasnet"><span>Download and load the pretrained Conv-Tasnet</span></a></h4><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span>!gdown --id 17DMWdw84wF3fz3t7ia1zssdzhkpVQGZm -O /content/chime_tasnet_singlechannel.zip</span></span>
<span class="line"><span>!unzip /content/chime_tasnet_singlechannel.zip -d /content/enh_model_sc</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div></div></div><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span># Load the model</span></span>
<span class="line"><span># If you encounter error &quot;No module named &#39;espnet2&#39;&quot;, please re-run the 1st Cell. This might be a colab bug.</span></span>
<span class="line"><span>import sys</span></span>
<span class="line"><span>import soundfile</span></span>
<span class="line"><span>from espnet2.bin.enh_inference import SeparateSpeech</span></span>
<span class="line"><span></span></span>
<span class="line"><span></span></span>
<span class="line"><span>separate_speech = {}</span></span>
<span class="line"><span># For models downloaded from GoogleDrive, you can use the following script:</span></span>
<span class="line"><span>enh_model_sc = SeparateSpeech(</span></span>
<span class="line"><span>  train_config=&quot;/content/enh_model_sc/exp/enh_train_enh_conv_tasnet_raw/config.yaml&quot;,</span></span>
<span class="line"><span>  model_file=&quot;/content/enh_model_sc/exp/enh_train_enh_conv_tasnet_raw/5epoch.pth&quot;,</span></span>
<span class="line"><span>  # for segment-wise process on long speech</span></span>
<span class="line"><span>  normalize_segment_scale=False,</span></span>
<span class="line"><span>  show_progressbar=True,</span></span>
<span class="line"><span>  ref_channel=4,</span></span>
<span class="line"><span>  normalize_output_wav=True,</span></span>
<span class="line"><span>  device=&quot;cuda:0&quot;,</span></span>
<span class="line"><span>)</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h4 id="enhance-the-single-channel-real-noisy-speech-in-chime4" tabindex="-1"><a class="header-anchor" href="#enhance-the-single-channel-real-noisy-speech-in-chime4"><span>Enhance the single-channel real noisy speech in CHiME4</span></a></h4><p>Please submit the screenshot of output of current block and the spectogram and waveform of noisy and enhanced speech file to Gradescope for Task 1.</p><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span># play the enhanced single-channel speech</span></span>
<span class="line"><span>wave = enh_model_sc(mixwav_sc[None, ...], sr)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>print(&quot;Input real noisy speech&quot;, flush=True)</span></span>
<span class="line"><span>display(Audio(mixwav_sc, rate=sr))</span></span>
<span class="line"><span>print(&quot;Enhanced speech&quot;, flush=True)</span></span>
<span class="line"><span>display(Audio(wave[0].squeeze(), rate=sr))</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="multi-channel-enhancement" tabindex="-1"><a class="header-anchor" href="#multi-channel-enhancement"><span>Multi-Channel Enhancement</span></a></h3><h4 id="download-and-load-the-pretrained-mvdr-neural-beamformer" tabindex="-1"><a class="header-anchor" href="#download-and-load-the-pretrained-mvdr-neural-beamformer"><span>Download and load the pretrained mvdr neural beamformer.</span></a></h4><h3 id="task2-✅-checkpoint-2-1-point" tabindex="-1"><a class="header-anchor" href="#task2-✅-checkpoint-2-1-point"><span>Task2 (✅ Checkpoint 2 (1 point))</span></a></h3><p>Run inference of pretrained multi-channel enhancement model.</p><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span># Download the pretained enhancement model</span></span>
<span class="line"><span></span></span>
<span class="line"><span>!gdown --id 1FohDfBlOa7ipc9v2luY-QIFQ_GJ1iW_i -O /content/mvdr_beamformer_16k_se_raw_valid.zip</span></span>
<span class="line"><span>!unzip /content/mvdr_beamformer_16k_se_raw_valid.zip -d /content/enh_model_mc</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span># Load the model</span></span>
<span class="line"><span># If you encounter error &quot;No module named &#39;espnet2&#39;&quot;, please re-run the 1st Cell. This might be a colab bug.</span></span>
<span class="line"><span>import sys</span></span>
<span class="line"><span>import soundfile</span></span>
<span class="line"><span>from espnet2.bin.enh_inference import SeparateSpeech</span></span>
<span class="line"><span></span></span>
<span class="line"><span></span></span>
<span class="line"><span>separate_speech = {}</span></span>
<span class="line"><span># For models downloaded from GoogleDrive, you can use the following script:</span></span>
<span class="line"><span>enh_model_mc = SeparateSpeech(</span></span>
<span class="line"><span>  train_config=&quot;/content/enh_model_mc/exp/enh_train_enh_beamformer_mvdr_raw/config.yaml&quot;,</span></span>
<span class="line"><span>  model_file=&quot;/content/enh_model_mc/exp/enh_train_enh_beamformer_mvdr_raw/11epoch.pth&quot;,</span></span>
<span class="line"><span>  # for segment-wise process on long speech</span></span>
<span class="line"><span>  normalize_segment_scale=False,</span></span>
<span class="line"><span>  show_progressbar=True,</span></span>
<span class="line"><span>  ref_channel=4,</span></span>
<span class="line"><span>  normalize_output_wav=True,</span></span>
<span class="line"><span>  device=&quot;cuda:0&quot;,</span></span>
<span class="line"><span>)</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h4 id="enhance-the-multi-channel-real-noisy-speech-in-chime4" tabindex="-1"><a class="header-anchor" href="#enhance-the-multi-channel-real-noisy-speech-in-chime4"><span>Enhance the multi-channel real noisy speech in CHiME4</span></a></h4><p>Please submit the screenshot of output of current block and the spectrogram and waveform of noisy and enhanced speech file to Gradescope for Task 2.</p><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span>wave = enh_model_mc(mixwav_mc[None, ...], sr)</span></span>
<span class="line"><span>print(&quot;Input real noisy speech&quot;, flush=True)</span></span>
<span class="line"><span>display(Audio(mixwav_mc.T, rate=sr))</span></span>
<span class="line"><span>print(&quot;Enhanced speech&quot;, flush=True)</span></span>
<span class="line"><span>display(Audio(wave[0].squeeze(), rate=sr))</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h4 id="portable-speech-enhancement-scripts-for-other-tasks" tabindex="-1"><a class="header-anchor" href="#portable-speech-enhancement-scripts-for-other-tasks"><span>Portable speech enhancement scripts for other tasks</span></a></h4><p>For an ESPNet ASR or TTS dataset like below:</p><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span>data</span></span>
<span class="line"><span>\`-- et05_real_isolated_6ch_track</span></span>
<span class="line"><span>    |-- spk2utt</span></span>
<span class="line"><span>    |-- text</span></span>
<span class="line"><span>    |-- utt2spk</span></span>
<span class="line"><span>    |-- utt2uniq</span></span>
<span class="line"><span>    \`-- wav.scp</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>Run the following scripts to create an enhanced dataset:</p><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span>scripts/utils/enhance_dataset.sh \\</span></span>
<span class="line"><span>    --spk_num 1 \\</span></span>
<span class="line"><span>    --gpu_inference true \\</span></span>
<span class="line"><span>    --inference_nj 4 \\</span></span>
<span class="line"><span>    --fs 16k \\</span></span>
<span class="line"><span>    --id_prefix &quot;&quot; \\</span></span>
<span class="line"><span>    dump/raw/et05_real_isolated_6ch_track \\</span></span>
<span class="line"><span>    data/et05_real_isolated_6ch_track_enh \\</span></span>
<span class="line"><span>    exp/enh_train_enh_beamformer_mvdr_raw/valid.loss.best.pth</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>The above script will generate a new directory data/et05_real_isolated_6ch_track_enh:</p><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span>data</span></span>
<span class="line"><span>\`-- et05_real_isolated_6ch_track_enh</span></span>
<span class="line"><span>    |-- spk2utt</span></span>
<span class="line"><span>    |-- text</span></span>
<span class="line"><span>    |-- utt2spk</span></span>
<span class="line"><span>    |-- utt2uniq</span></span>
<span class="line"><span>    |-- wav.scp</span></span>
<span class="line"><span>    \`-- wavs/</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>where wav.scp contains paths to the enhanced audios (stored in wavs/).</p><h2 id="speech-separation" tabindex="-1"><a class="header-anchor" href="#speech-separation"><span>Speech Separation</span></a></h2><h3 id="model-selection" tabindex="-1"><a class="header-anchor" href="#model-selection"><span>Model Selection</span></a></h3><p>In this demonstration, we will show different speech separation models on wsj0_2mix.</p><p>The pretrained models can be download from a direct URL, or from <a href="https://zenodo.org/" target="_blank" rel="noopener noreferrer">zenodo</a> and <a href="https://huggingface.co/" target="_blank" rel="noopener noreferrer">huggingface</a> with the corresponding model ID.</p><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span>!gdown --id 1TasZxZSnbSPsk_Wf7ZDhBAigS6zN8G9G -O enh_train_enh_tfgridnet_tf_lr-patience3_patience5_raw_valid.loss.ave.zip</span></span>
<span class="line"><span>!unzip enh_train_enh_tfgridnet_tf_lr-patience3_patience5_raw_valid.loss.ave.zip -d /content/enh_model_ss</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div></div></div><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span>import sys</span></span>
<span class="line"><span>import soundfile</span></span>
<span class="line"><span>from espnet2.bin.enh_inference import SeparateSpeech</span></span>
<span class="line"><span></span></span>
<span class="line"><span># For models downloaded from GoogleDrive, you can use the following script:</span></span>
<span class="line"><span>separate_speech = SeparateSpeech(</span></span>
<span class="line"><span>  train_config=&quot;/content/enh_model_ss/exp/enh_train_enh_tfgridnet_tf_lr-patience3_patience5_raw/config.yaml&quot;,</span></span>
<span class="line"><span>  model_file=&quot;/content/enh_model_ss/exp/enh_train_enh_tfgridnet_tf_lr-patience3_patience5_raw/98epoch.pth&quot;,</span></span>
<span class="line"><span>  # for segment-wise process on long speech</span></span>
<span class="line"><span>  segment_size=2.4,</span></span>
<span class="line"><span>  hop_size=0.8,</span></span>
<span class="line"><span>  normalize_segment_scale=False,</span></span>
<span class="line"><span>  show_progressbar=True,</span></span>
<span class="line"><span>  ref_channel=None,</span></span>
<span class="line"><span>  normalize_output_wav=True,</span></span>
<span class="line"><span>  device=&quot;cuda:0&quot;,</span></span>
<span class="line"><span>)</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="separate-speech-mixture" tabindex="-1"><a class="header-anchor" href="#separate-speech-mixture"><span>Separate Speech Mixture</span></a></h3><h4 id="separate-the-example-in-wsj0-2mix-testing-set" tabindex="-1"><a class="header-anchor" href="#separate-the-example-in-wsj0-2mix-testing-set"><span>Separate the example in wsj0_2mix testing set</span></a></h4><h3 id="task3-✅-checkpoint-3-1-point" tabindex="-1"><a class="header-anchor" href="#task3-✅-checkpoint-3-1-point"><span>Task3 (✅ Checkpoint 3 (1 point))</span></a></h3><p>Run inference of pretrained speech seperation model based on TF-GRIDNET.</p><p>Please submit the screenshot of output of current block and the spectrogram and waveform of mixed and seperated speech files to Gradescope for Task 3.</p><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span>!gdown --id 1ZCUkd_Lb7pO2rpPr4FqYdtJBZ7JMiInx -O /content/447c020t_1.2106_422a0112_-1.2106.wav</span></span>
<span class="line"><span></span></span>
<span class="line"><span>import os</span></span>
<span class="line"><span>import soundfile</span></span>
<span class="line"><span>from IPython.display import display, Audio</span></span>
<span class="line"><span></span></span>
<span class="line"><span>mixwav, sr = soundfile.read(&quot;447c020t_1.2106_422a0112_-1.2106.wav&quot;)</span></span>
<span class="line"><span>waves_wsj = separate_speech(mixwav[None, ...], fs=sr)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>print(&quot;Input mixture&quot;, flush=True)</span></span>
<span class="line"><span>display(Audio(mixwav, rate=sr))</span></span>
<span class="line"><span>print(f&quot;========= Separated speech with model =========&quot;, flush=True)</span></span>
<span class="line"><span>print(&quot;Separated spk1&quot;, flush=True)</span></span>
<span class="line"><span>display(Audio(waves_wsj[0].squeeze(), rate=sr))</span></span>
<span class="line"><span>print(&quot;Separated spk2&quot;, flush=True)</span></span>
<span class="line"><span>display(Audio(waves_wsj[1].squeeze(), rate=sr))</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h4 id="show-spectrums-of-separated-speech" tabindex="-1"><a class="header-anchor" href="#show-spectrums-of-separated-speech"><span>Show spectrums of separated speech</span></a></h4><p>Show wavform and spectrogram of mixed and seperated speech.</p><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span>import matplotlib.pyplot as plt</span></span>
<span class="line"><span>import torch</span></span>
<span class="line"><span>from torch_complex.tensor import ComplexTensor</span></span>
<span class="line"><span></span></span>
<span class="line"><span>from espnet.asr.asr_utils import plot_spectrogram</span></span>
<span class="line"><span>from espnet2.layers.stft import Stft</span></span>
<span class="line"><span></span></span>
<span class="line"><span></span></span>
<span class="line"><span>stft = Stft(</span></span>
<span class="line"><span>  n_fft=512,</span></span>
<span class="line"><span>  win_length=None,</span></span>
<span class="line"><span>  hop_length=128,</span></span>
<span class="line"><span>  window=&quot;hann&quot;,</span></span>
<span class="line"><span>)</span></span>
<span class="line"><span>ilens = torch.LongTensor([len(mixwav)])</span></span>
<span class="line"><span># specs: (T, F)</span></span>
<span class="line"><span>spec_mix = ComplexTensor(</span></span>
<span class="line"><span>    *torch.unbind(</span></span>
<span class="line"><span>      stft(torch.as_tensor(mixwav).unsqueeze(0), ilens)[0].squeeze(),</span></span>
<span class="line"><span>      dim=-1</span></span>
<span class="line"><span>  )</span></span>
<span class="line"><span>)</span></span>
<span class="line"><span>spec_sep1 = ComplexTensor(</span></span>
<span class="line"><span>    *torch.unbind(</span></span>
<span class="line"><span>      stft(torch.as_tensor(waves_wsj[0]), ilens)[0].squeeze(),</span></span>
<span class="line"><span>      dim=-1</span></span>
<span class="line"><span>  )</span></span>
<span class="line"><span>)</span></span>
<span class="line"><span>spec_sep2 = ComplexTensor(</span></span>
<span class="line"><span>    *torch.unbind(</span></span>
<span class="line"><span>      stft(torch.as_tensor(waves_wsj[1]), ilens)[0].squeeze(),</span></span>
<span class="line"><span>      dim=-1</span></span>
<span class="line"><span>  )</span></span>
<span class="line"><span>)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>samples = torch.linspace(0, len(mixwav) / sr, len(mixwav))</span></span>
<span class="line"><span>plt.figure(figsize=(24, 12))</span></span>
<span class="line"><span>plt.subplot(3, 2, 1)</span></span>
<span class="line"><span>plt.title(&#39;Mixture Spectrogram&#39;)</span></span>
<span class="line"><span>plot_spectrogram(</span></span>
<span class="line"><span>  plt, abs(spec_mix).transpose(-1, -2).numpy(), fs=sr,</span></span>
<span class="line"><span>  mode=&#39;db&#39;, frame_shift=None,</span></span>
<span class="line"><span>  bottom=False, labelbottom=False</span></span>
<span class="line"><span>)</span></span>
<span class="line"><span>plt.subplot(3, 2, 2)</span></span>
<span class="line"><span>plt.title(&#39;Mixture Wavform&#39;)</span></span>
<span class="line"><span>plt.plot(samples, mixwav)</span></span>
<span class="line"><span>plt.xlim(0, len(mixwav) / sr)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>plt.subplot(3, 2, 3)</span></span>
<span class="line"><span>plt.title(&#39;Separated Spectrogram (spk1)&#39;)</span></span>
<span class="line"><span>plot_spectrogram(</span></span>
<span class="line"><span>  plt, abs(spec_sep1).transpose(-1, -2).numpy(), fs=sr,</span></span>
<span class="line"><span>  mode=&#39;db&#39;, frame_shift=None,</span></span>
<span class="line"><span>  bottom=False, labelbottom=False</span></span>
<span class="line"><span>)</span></span>
<span class="line"><span>plt.subplot(3, 2, 4)</span></span>
<span class="line"><span>plt.title(&#39;Separated Wavform (spk1)&#39;)</span></span>
<span class="line"><span>plt.plot(samples, waves_wsj[0].squeeze())</span></span>
<span class="line"><span>plt.xlim(0, len(mixwav) / sr)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>plt.subplot(3, 2, 5)</span></span>
<span class="line"><span>plt.title(&#39;Separated Spectrogram (spk2)&#39;)</span></span>
<span class="line"><span>plot_spectrogram(</span></span>
<span class="line"><span>  plt, abs(spec_sep2).transpose(-1, -2).numpy(), fs=sr,</span></span>
<span class="line"><span>  mode=&#39;db&#39;, frame_shift=None,</span></span>
<span class="line"><span>  bottom=False, labelbottom=False</span></span>
<span class="line"><span>)</span></span>
<span class="line"><span>plt.subplot(3, 2, 6)</span></span>
<span class="line"><span>plt.title(&#39;Separated Wavform (spk2)&#39;)</span></span>
<span class="line"><span>plt.plot(samples, waves_wsj[1].squeeze())</span></span>
<span class="line"><span>plt.xlim(0, len(mixwav) / sr)</span></span>
<span class="line"><span>plt.xlabel(&quot;Time (s)&quot;)</span></span>
<span class="line"><span>plt.show()</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h2 id="evaluate-separated-speech-with-pretrained-asr-model" tabindex="-1"><a class="header-anchor" href="#evaluate-separated-speech-with-pretrained-asr-model"><span>Evaluate separated speech with pretrained ASR model</span></a></h2><p>The ground truths are:</p><p><code>text_1: SOME CRITICS INCLUDING HIGH REAGAN ADMINISTRATION OFFICIALS ARE RAISING THE ALARM THAT THE FED&#39;S POLICY IS TOO TIGHT AND COULD CAUSE A RECESSION NEXT YEAR</code></p><p><code>text_2: THE UNITED STATES UNDERTOOK TO DEFEND WESTERN EUROPE AGAINST SOVIET ATTACK</code></p><p>(This may take a while for the speech recognition.)</p><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span>%pip install -q https://github.com/kpu/kenlm/archive/master.zip # ASR needs kenlm</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div></div></div><h3 id="task4-✅-checkpoint-4-1-point" tabindex="-1"><a class="header-anchor" href="#task4-✅-checkpoint-4-1-point"><span>Task4 (✅ Checkpoint 4 (1 point))</span></a></h3><p>Show inference of pre-trained ASR model on mixed and seperated speech.</p><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span>!gdown --id 1H7--jXTTwmwxzfO8LT5kjZyBjng-HxED -O asr_train_asr_transformer_raw_char_1gpu_valid.acc.ave.zip</span></span>
<span class="line"><span>!unzip asr_train_asr_transformer_raw_char_1gpu_valid.acc.ave.zip -d /content/asr_model</span></span>
<span class="line"><span>!ln -sf /content/asr_model/exp .</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>Please submit the screenshot of ASR inference on Mix Speech and Separated Speech 1 and Separated Speech 2 files to Gradescope for Task 4.</p><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span>import espnet_model_zoo</span></span>
<span class="line"><span>from espnet2.bin.asr_inference import Speech2Text</span></span>
<span class="line"><span></span></span>
<span class="line"><span></span></span>
<span class="line"><span># For models downloaded from GoogleDrive, you can use the following script:</span></span>
<span class="line"><span>speech2text = Speech2Text(</span></span>
<span class="line"><span>  asr_train_config=&quot;/content/asr_model/exp/asr_train_asr_transformer_raw_char_1gpu/config.yaml&quot;,</span></span>
<span class="line"><span>  asr_model_file=&quot;/content/asr_model/exp/asr_train_asr_transformer_raw_char_1gpu/valid.acc.ave_10best.pth&quot;,</span></span>
<span class="line"><span>  device=&quot;cuda:0&quot;</span></span>
<span class="line"><span>)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>text_est = [None, None]</span></span>
<span class="line"><span>text_est[0], *_ = speech2text(waves_wsj[0].squeeze())[0]</span></span>
<span class="line"><span>text_est[1], *_ = speech2text(waves_wsj[1].squeeze())[0]</span></span>
<span class="line"><span>text_m, *_ = speech2text(mixwav)[0]</span></span>
<span class="line"><span>print(&quot;Mix Speech to Text: &quot;, text_m)</span></span>
<span class="line"><span>print(&quot;Separated Speech 1 to Text: &quot;, text_est[0])</span></span>
<span class="line"><span>print(&quot;Separated Speech 2 to Text: &quot;, text_est[1])</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span>import difflib</span></span>
<span class="line"><span>from itertools import permutations</span></span>
<span class="line"><span></span></span>
<span class="line"><span>import editdistance</span></span>
<span class="line"><span>import numpy as np</span></span>
<span class="line"><span></span></span>
<span class="line"><span>colors = dict(</span></span>
<span class="line"><span>    red=lambda text: f&quot;\\033[38;2;255;0;0m{text}\\033[0m&quot; if text else &quot;&quot;,</span></span>
<span class="line"><span>    green=lambda text: f&quot;\\033[38;2;0;255;0m{text}\\033[0m&quot; if text else &quot;&quot;,</span></span>
<span class="line"><span>    yellow=lambda text: f&quot;\\033[38;2;225;225;0m{text}\\033[0m&quot; if text else &quot;&quot;,</span></span>
<span class="line"><span>    white=lambda text: f&quot;\\033[38;2;255;255;255m{text}\\033[0m&quot; if text else &quot;&quot;,</span></span>
<span class="line"><span>    black=lambda text: f&quot;\\033[38;2;0;0;0m{text}\\033[0m&quot; if text else &quot;&quot;,</span></span>
<span class="line"><span>)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>def diff_strings(ref, est):</span></span>
<span class="line"><span>    &quot;&quot;&quot;Reference: https://stackoverflow.com/a/64404008/7384873&quot;&quot;&quot;</span></span>
<span class="line"><span>    ref_str, est_str, err_str = [], [], []</span></span>
<span class="line"><span>    matcher = difflib.SequenceMatcher(None, ref, est)</span></span>
<span class="line"><span>    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():</span></span>
<span class="line"><span>        if opcode == &quot;equal&quot;:</span></span>
<span class="line"><span>            txt = ref[a0:a1]</span></span>
<span class="line"><span>            ref_str.append(txt)</span></span>
<span class="line"><span>            est_str.append(txt)</span></span>
<span class="line"><span>            err_str.append(&quot; &quot; * (a1 - a0))</span></span>
<span class="line"><span>        elif opcode == &quot;insert&quot;:</span></span>
<span class="line"><span>            ref_str.append(&quot;*&quot; * (b1 - b0))</span></span>
<span class="line"><span>            est_str.append(colors[&quot;green&quot;](est[b0:b1]))</span></span>
<span class="line"><span>            err_str.append(colors[&quot;black&quot;](&quot;I&quot; * (b1 - b0)))</span></span>
<span class="line"><span>        elif opcode == &quot;delete&quot;:</span></span>
<span class="line"><span>            ref_str.append(ref[a0:a1])</span></span>
<span class="line"><span>            est_str.append(colors[&quot;red&quot;](&quot;*&quot; * (a1 - a0)))</span></span>
<span class="line"><span>            err_str.append(colors[&quot;black&quot;](&quot;D&quot; * (a1 - a0)))</span></span>
<span class="line"><span>        elif opcode == &quot;replace&quot;:</span></span>
<span class="line"><span>            diff = a1 - a0 - b1 + b0</span></span>
<span class="line"><span>            if diff &gt;= 0:</span></span>
<span class="line"><span>                txt_ref = ref[a0:a1]</span></span>
<span class="line"><span>                txt_est = colors[&quot;yellow&quot;](est[b0:b1]) + colors[&quot;red&quot;](&quot;*&quot; * diff)</span></span>
<span class="line"><span>                txt_err = &quot;S&quot; * (b1 - b0) + &quot;D&quot; * diff</span></span>
<span class="line"><span>            elif diff &lt; 0:</span></span>
<span class="line"><span>                txt_ref = ref[a0:a1] + &quot;*&quot; * -diff</span></span>
<span class="line"><span>                txt_est = colors[&quot;yellow&quot;](est[b0:b1]) + colors[&quot;green&quot;](&quot;*&quot; * -diff)</span></span>
<span class="line"><span>                txt_err = &quot;S&quot; * (b1 - b0) + &quot;I&quot; * -diff</span></span>
<span class="line"><span></span></span>
<span class="line"><span>            ref_str.append(txt_ref)</span></span>
<span class="line"><span>            est_str.append(txt_est)</span></span>
<span class="line"><span>            err_str.append(colors[&quot;black&quot;](txt_err))</span></span>
<span class="line"><span>    return &quot;&quot;.join(ref_str), &quot;&quot;.join(est_str), &quot;&quot;.join(err_str)</span></span>
<span class="line"><span></span></span>
<span class="line"><span></span></span>
<span class="line"><span>text_ref = [</span></span>
<span class="line"><span>  &quot;SOME CRITICS INCLUDING HIGH REAGAN ADMINISTRATION OFFICIALS ARE RAISING THE ALARM THAT THE FED&#39;S POLICY IS TOO TIGHT AND COULD CAUSE A RECESSION NEXT YEAR&quot;,</span></span>
<span class="line"><span>  &quot;THE UNITED STATES UNDERTOOK TO DEFEND WESTERN EUROPE AGAINST SOVIET ATTACK&quot;,</span></span>
<span class="line"><span>]</span></span>
<span class="line"><span></span></span>
<span class="line"><span>print(&quot;=====================&quot; , flush=True)</span></span>
<span class="line"><span>perms = list(permutations(range(2)))</span></span>
<span class="line"><span>string_edit = [</span></span>
<span class="line"><span>  [</span></span>
<span class="line"><span>    editdistance.eval(text_ref[m], text_est[n])</span></span>
<span class="line"><span>    for m, n in enumerate(p)</span></span>
<span class="line"><span>  ]</span></span>
<span class="line"><span>  for p in perms</span></span>
<span class="line"><span>]</span></span>
<span class="line"><span></span></span>
<span class="line"><span>dist = [sum(edist) for edist in string_edit]</span></span>
<span class="line"><span>perm_idx = np.argmin(dist)</span></span>
<span class="line"><span>perm = perms[perm_idx]</span></span>
<span class="line"><span></span></span>
<span class="line"><span>for i, p in enumerate(perm):</span></span>
<span class="line"><span>  print(&quot;\\n--------------- Text %d ---------------&quot; % (i + 1), flush=True)</span></span>
<span class="line"><span>  ref, est, err = diff_strings(text_ref[i], text_est[p])</span></span>
<span class="line"><span>  print(&quot;REF: &quot; + ref + &quot;\\n&quot; + &quot;HYP: &quot; + est + &quot;\\n&quot; + &quot;ERR: &quot; + err, flush=True)</span></span>
<span class="line"><span>  print(&quot;Edit Distance = {}\\n&quot;.format(string_edit[perm_idx][i]), flush=True)</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="task5-✅-checkpoint-5-1-point" tabindex="-1"><a class="header-anchor" href="#task5-✅-checkpoint-5-1-point"><span>Task5 (✅ Checkpoint 5 (1 point))</span></a></h3><p>Enhance your own pre-recordings. Your input speech can be recorded by yourself or you can also find it from other sources (e.g., youtube).</p><p>Discuss whether input speech was clearly denoised, and if not, what would be a potential reason.</p><p>[YOUR ANSWER HERE]</p><p>Please submit the spectrogram and waveform of your input and enhanced speech to GradeScope for Task 5 along with the screenshot of your answer.</p><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span>from google.colab import files</span></span>
<span class="line"><span>from IPython.display import display, Audio</span></span>
<span class="line"><span>import soundfile</span></span>
<span class="line"><span>fs = 16000 </span></span>
<span class="line"><span>uploaded = files.upload()</span></span>
<span class="line"><span></span></span>
<span class="line"><span>for file_name in uploaded.keys():</span></span>
<span class="line"><span>  speech, rate = soundfile.read(file_name)</span></span>
<span class="line"><span>  assert rate == fs, &quot;mismatch in sampling rate&quot;</span></span>
<span class="line"><span>  wave = enh_model_sc(speech[None, ...], fs)</span></span>
<span class="line"><span>  print(f&quot;Your input speech {file_name}&quot;, flush=True)</span></span>
<span class="line"><span>  display(Audio(speech, rate=fs))</span></span>
<span class="line"><span>  print(f&quot;Enhanced speech for {file_name}&quot;, flush=True)</span></span>
<span class="line"><span>  display(Audio(wave[0].squeeze(), rate=fs))</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>`,85),p=[l];function t(r,d){return e(),n("div",null,p)}const h=s(i,[["render",t],["__file","assignment7_se.html.vue"]]),u=JSON.parse('{"path":"/notebook/ESPnet2/Course/CMU_SpeechProcessing_Spring2023/assignment7_se.html","title":"CMU 11492/11692 Spring 2023: Speech Enhancement","lang":"en-US","frontmatter":{},"headers":[{"level":2,"title":"❗Important Notes❗","slug":"❗important-notes❗","link":"#❗important-notes❗","children":[]},{"level":2,"title":"Install","slug":"install","link":"#install","children":[]},{"level":2,"title":"Speech Enhancement with Pretrained Models","slug":"speech-enhancement-with-pretrained-models","link":"#speech-enhancement-with-pretrained-models","children":[{"level":3,"title":"Single-Channel Enhancement, the CHiME example","slug":"single-channel-enhancement-the-chime-example","link":"#single-channel-enhancement-the-chime-example","children":[]},{"level":3,"title":"Task1  (✅ Checkpoint 1 (1 point))","slug":"task1-✅-checkpoint-1-1-point","link":"#task1-✅-checkpoint-1-1-point","children":[]},{"level":3,"title":"Multi-Channel Enhancement","slug":"multi-channel-enhancement","link":"#multi-channel-enhancement","children":[]},{"level":3,"title":"Task2  (✅ Checkpoint 2 (1 point))","slug":"task2-✅-checkpoint-2-1-point","link":"#task2-✅-checkpoint-2-1-point","children":[]}]},{"level":2,"title":"Speech Separation","slug":"speech-separation","link":"#speech-separation","children":[{"level":3,"title":"Model Selection","slug":"model-selection","link":"#model-selection","children":[]},{"level":3,"title":"Separate Speech Mixture","slug":"separate-speech-mixture","link":"#separate-speech-mixture","children":[]},{"level":3,"title":"Task3  (✅ Checkpoint 3 (1 point))","slug":"task3-✅-checkpoint-3-1-point","link":"#task3-✅-checkpoint-3-1-point","children":[]}]},{"level":2,"title":"Evaluate separated speech with pretrained ASR model","slug":"evaluate-separated-speech-with-pretrained-asr-model","link":"#evaluate-separated-speech-with-pretrained-asr-model","children":[{"level":3,"title":"Task4  (✅ Checkpoint 4 (1 point))","slug":"task4-✅-checkpoint-4-1-point","link":"#task4-✅-checkpoint-4-1-point","children":[]},{"level":3,"title":"Task5  (✅ Checkpoint 5 (1 point))","slug":"task5-✅-checkpoint-5-1-point","link":"#task5-✅-checkpoint-5-1-point","children":[]}]}],"git":{"createdTime":null,"updatedTime":null,"contributors":[]},"readingTime":{"minutes":6.4,"words":1920},"filePathRelative":"notebook/ESPnet2/Course/CMU_SpeechProcessing_Spring2023/assignment7_se.md","excerpt":"\\n<p>In this demonstration, we will show you some demonstrations of speech enhancement systems in ESPnet.</p>\\n<p>Main references:</p>\\n<ul>\\n<li><a href=\\"https://github.com/espnet/espnet\\" target=\\"_blank\\" rel=\\"noopener noreferrer\\">ESPnet repository</a></li>\\n<li><a href=\\"https://espnet.github.io/espnet/\\" target=\\"_blank\\" rel=\\"noopener noreferrer\\">ESPnet documentation</a></li>\\n<li><a href=\\"https://github.com/espnet/espnet/tree/master/egs2/TEMPLATE/enh1\\" target=\\"_blank\\" rel=\\"noopener noreferrer\\">ESPnet-SE repo</a></li>\\n</ul>"}');export{h as comp,u as data};
