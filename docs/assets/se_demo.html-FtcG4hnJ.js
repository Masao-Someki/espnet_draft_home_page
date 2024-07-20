import{_ as s}from"./plugin-vue_export-helper-DlAUqK2U.js";import{c as n,o as a,a as e}from"./app-KOUU_Wij.js";const i={},l=e(`<h1 id="espnet-speech-enhancement-demonstration" tabindex="-1"><a class="header-anchor" href="#espnet-speech-enhancement-demonstration"><span>ESPnet Speech Enhancement Demonstration</span></a></h1><figure><a href="https://colab.research.google.com/drive/1fjRJCh96SoYLZPRxsjF9VDv4Q2VoIckI?usp=sharing" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" tabindex="0" loading="lazy"></a><figcaption>Open In Colab</figcaption></figure><p>This notebook provides a demonstration of the speech enhancement and separation using ESPnet2-SE.</p><ul><li>ESPnet2-SE: https://github.com/espnet/espnet/tree/master/egs2/TEMPLATE/enh1</li></ul><p>Author: Chenda Li (<a href="https://github.com/LiChenda" target="_blank" rel="noopener noreferrer">@LiChenda</a>), Wangyou Zhang (<a href="https://github.com/Emrys365" target="_blank" rel="noopener noreferrer">@Emrys365</a>)</p><h2 id="install" tabindex="-1"><a class="header-anchor" href="#install"><span>Install</span></a></h2><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span>%pip install -q espnet==0.10.1</span></span>
<span class="line"><span>%pip install -q espnet_model_zoo</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div></div></div><h2 id="speech-enhancement" tabindex="-1"><a class="header-anchor" href="#speech-enhancement"><span>Speech Enhancement</span></a></h2><h3 id="single-channel-enhancement-the-chime-example" tabindex="-1"><a class="header-anchor" href="#single-channel-enhancement-the-chime-example"><span>Single-Channel Enhancement, the CHiME example</span></a></h3><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span># Download one utterance from real noisy speech of CHiME4</span></span>
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
<span class="line"><span>  enh_train_config=&quot;/content/enh_model_sc/exp/enh_train_enh_conv_tasnet_raw/config.yaml&quot;,</span></span>
<span class="line"><span>  enh_model_file=&quot;/content/enh_model_sc/exp/enh_train_enh_conv_tasnet_raw/5epoch.pth&quot;,</span></span>
<span class="line"><span>  # for segment-wise process on long speech</span></span>
<span class="line"><span>  normalize_segment_scale=False,</span></span>
<span class="line"><span>  show_progressbar=True,</span></span>
<span class="line"><span>  ref_channel=4,</span></span>
<span class="line"><span>  normalize_output_wav=True,</span></span>
<span class="line"><span>  device=&quot;cuda:0&quot;,</span></span>
<span class="line"><span>)</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h4 id="enhance-the-single-channel-real-noisy-speech-in-chime4" tabindex="-1"><a class="header-anchor" href="#enhance-the-single-channel-real-noisy-speech-in-chime4"><span>Enhance the single-channel real noisy speech in CHiME4</span></a></h4><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span># play the enhanced single-channel speech</span></span>
<span class="line"><span>wave = enh_model_sc(mixwav_sc[None, ...], sr)</span></span>
<span class="line"><span>print(&quot;Input real noisy speech&quot;, flush=True)</span></span>
<span class="line"><span>display(Audio(mixwav_sc, rate=sr))</span></span>
<span class="line"><span>print(&quot;Enhanced speech&quot;, flush=True)</span></span>
<span class="line"><span>display(Audio(wave[0].squeeze(), rate=sr))</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="enhance-your-own-pre-recordings" tabindex="-1"><a class="header-anchor" href="#enhance-your-own-pre-recordings"><span>Enhance your own pre-recordings</span></a></h3><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span>from google.colab import files</span></span>
<span class="line"><span>from IPython.display import display, Audio</span></span>
<span class="line"><span>import soundfile</span></span>
<span class="line"><span></span></span>
<span class="line"><span>uploaded = files.upload()</span></span>
<span class="line"><span></span></span>
<span class="line"><span>for file_name in uploaded.keys():</span></span>
<span class="line"><span>  speech, rate = soundfile.read(file_name)</span></span>
<span class="line"><span>  assert rate == sr, &quot;mismatch in sampling rate&quot;</span></span>
<span class="line"><span>  wave = enh_model_sc(speech[None, ...], sr)</span></span>
<span class="line"><span>  print(f&quot;Your input speech {file_name}&quot;, flush=True)</span></span>
<span class="line"><span>  display(Audio(speech, rate=sr))</span></span>
<span class="line"><span>  print(f&quot;Enhanced speech for {file_name}&quot;, flush=True)</span></span>
<span class="line"><span>  display(Audio(wave[0].squeeze(), rate=sr))</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="multi-channel-enhancement" tabindex="-1"><a class="header-anchor" href="#multi-channel-enhancement"><span>Multi-Channel Enhancement</span></a></h3><h4 id="download-and-load-the-pretrained-mvdr-neural-beamformer" tabindex="-1"><a class="header-anchor" href="#download-and-load-the-pretrained-mvdr-neural-beamformer"><span>Download and load the pretrained mvdr neural beamformer.</span></a></h4><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span># Download the pretained enhancement model</span></span>
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
<span class="line"><span>  enh_train_config=&quot;/content/enh_model_mc/exp/enh_train_enh_beamformer_mvdr_raw/config.yaml&quot;,</span></span>
<span class="line"><span>  enh_model_file=&quot;/content/enh_model_mc/exp/enh_train_enh_beamformer_mvdr_raw/11epoch.pth&quot;,</span></span>
<span class="line"><span>  # for segment-wise process on long speech</span></span>
<span class="line"><span>  normalize_segment_scale=False,</span></span>
<span class="line"><span>  show_progressbar=True,</span></span>
<span class="line"><span>  ref_channel=4,</span></span>
<span class="line"><span>  normalize_output_wav=True,</span></span>
<span class="line"><span>  device=&quot;cuda:0&quot;,</span></span>
<span class="line"><span>)</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h4 id="enhance-the-multi-channel-real-noisy-speech-in-chime4" tabindex="-1"><a class="header-anchor" href="#enhance-the-multi-channel-real-noisy-speech-in-chime4"><span>Enhance the multi-channel real noisy speech in CHiME4</span></a></h4><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span>wave = enh_model_mc(mixwav_mc[None, ...], sr)</span></span>
<span class="line"><span>print(&quot;Input real noisy speech&quot;, flush=True)</span></span>
<span class="line"><span>display(Audio(mixwav_mc.T, rate=sr))</span></span>
<span class="line"><span>print(&quot;Enhanced speech&quot;, flush=True)</span></span>
<span class="line"><span>display(Audio(wave[0].squeeze(), rate=sr))</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h2 id="speech-separation" tabindex="-1"><a class="header-anchor" href="#speech-separation"><span>Speech Separation</span></a></h2><h3 id="model-selection" tabindex="-1"><a class="header-anchor" href="#model-selection"><span>Model Selection</span></a></h3><p>Please select model shown in <a href="https://github.com/espnet/espnet_model_zoo/blob/master/espnet_model_zoo/table.csv" target="_blank" rel="noopener noreferrer">espnet_model_zoo</a></p><p>In this demonstration, we will show different speech separation models on wsj0_2mix.</p><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span>#@title Choose Speech Separation model { run: &quot;auto&quot; }</span></span>
<span class="line"><span></span></span>
<span class="line"><span>fs = 8000 #@param {type:&quot;integer&quot;}</span></span>
<span class="line"><span>tag = &quot;Chenda Li/wsj0_2mix_enh_train_enh_conv_tasnet_raw_valid.si_snr.ave&quot; #@param [&quot;Chenda Li/wsj0_2mix_enh_train_enh_conv_tasnet_raw_valid.si_snr.ave&quot;, &quot;Chenda Li/wsj0_2mix_enh_train_enh_rnn_tf_raw_valid.si_snr.ave&quot;, &quot;https://zenodo.org/record/4688000/files/enh_train_enh_dprnn_tasnet_raw_valid.si_snr.ave.zip&quot;]</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span># For models uploaded to Zenodo, you can use the following python script instead:</span></span>
<span class="line"><span>import sys</span></span>
<span class="line"><span>import soundfile</span></span>
<span class="line"><span>from espnet_model_zoo.downloader import ModelDownloader</span></span>
<span class="line"><span>from espnet2.bin.enh_inference import SeparateSpeech</span></span>
<span class="line"><span></span></span>
<span class="line"><span>d = ModelDownloader()</span></span>
<span class="line"><span></span></span>
<span class="line"><span>cfg = d.download_and_unpack(tag)</span></span>
<span class="line"><span>separate_speech = SeparateSpeech(</span></span>
<span class="line"><span>  enh_train_config=cfg[&quot;train_config&quot;],</span></span>
<span class="line"><span>  enh_model_file=cfg[&quot;model_file&quot;],</span></span>
<span class="line"><span>  # for segment-wise process on long speech</span></span>
<span class="line"><span>  segment_size=2.4,</span></span>
<span class="line"><span>  hop_size=0.8,</span></span>
<span class="line"><span>  normalize_segment_scale=False,</span></span>
<span class="line"><span>  show_progressbar=True,</span></span>
<span class="line"><span>  ref_channel=None,</span></span>
<span class="line"><span>  normalize_output_wav=True,</span></span>
<span class="line"><span>  device=&quot;cuda:0&quot;,</span></span>
<span class="line"><span>)</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="separate-speech-mixture" tabindex="-1"><a class="header-anchor" href="#separate-speech-mixture"><span>Separate Speech Mixture</span></a></h3><h4 id="separate-the-example-in-wsj0-2mix-testing-set" tabindex="-1"><a class="header-anchor" href="#separate-the-example-in-wsj0-2mix-testing-set"><span>Separate the example in wsj0_2mix testing set</span></a></h4><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span>!gdown --id 1ZCUkd_Lb7pO2rpPr4FqYdtJBZ7JMiInx -O /content/447c020t_1.2106_422a0112_-1.2106.wav</span></span>
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
<span class="line"><span>print(f&quot;========= Separated speech with model {tag} =========&quot;, flush=True)</span></span>
<span class="line"><span>print(&quot;Separated spk1&quot;, flush=True)</span></span>
<span class="line"><span>display(Audio(waves_wsj[0].squeeze(), rate=sr))</span></span>
<span class="line"><span>print(&quot;Separated spk2&quot;, flush=True)</span></span>
<span class="line"><span>display(Audio(waves_wsj[1].squeeze(), rate=sr))</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h4 id="separate-your-own-recordings" tabindex="-1"><a class="header-anchor" href="#separate-your-own-recordings"><span>Separate your own recordings</span></a></h4><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span>from google.colab import files</span></span>
<span class="line"><span>from IPython.display import display, Audio</span></span>
<span class="line"><span>import soundfile</span></span>
<span class="line"><span></span></span>
<span class="line"><span>uploaded = files.upload()</span></span>
<span class="line"><span></span></span>
<span class="line"><span>for file_name in uploaded.keys():</span></span>
<span class="line"><span>  mixwav_yours, rate = soundfile.read(file_name)</span></span>
<span class="line"><span>  assert rate == sr, &quot;mismatch in sampling rate&quot;</span></span>
<span class="line"><span>  waves_yours = separate_speech(mixwav_yours[None, ...], fs=sr)</span></span>
<span class="line"><span>  print(&quot;Input mixture&quot;, flush=True)</span></span>
<span class="line"><span>  display(Audio(mixwav_yours, rate=sr))</span></span>
<span class="line"><span>  print(f&quot;========= Separated speech with model {tag} =========&quot;, flush=True)</span></span>
<span class="line"><span>  print(&quot;Separated spk1&quot;, flush=True)</span></span>
<span class="line"><span>  display(Audio(waves_yours[0].squeeze(), rate=sr))</span></span>
<span class="line"><span>  print(&quot;Separated spk2&quot;, flush=True)</span></span>
<span class="line"><span>  display(Audio(waves_yours[1].squeeze(), rate=sr))</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h4 id="show-spectrums-of-separated-speech" tabindex="-1"><a class="header-anchor" href="#show-spectrums-of-separated-speech"><span>Show spectrums of separated speech</span></a></h4><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span>import matplotlib.pyplot as plt</span></span>
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
<span class="line"><span># freqs = torch.linspace(0, sr / 2, spec_mix.shape[1])</span></span>
<span class="line"><span># frames = torch.linspace(0, len(mixwav) / sr, spec_mix.shape[0])</span></span>
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
<span class="line"><span>plt.show()</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h2 id="evluate-separated-speech-with-pretrained-asr-model" tabindex="-1"><a class="header-anchor" href="#evluate-separated-speech-with-pretrained-asr-model"><span>Evluate separated speech with pretrained ASR model</span></a></h2><p>The ground truths are:</p><p><code>text_1: SOME CRITICS INCLUDING HIGH REAGAN ADMINISTRATION OFFICIALS ARE RAISING THE ALARM THAT THE FED&#39;S POLICY IS TOO TIGHT AND COULD CAUSE A RECESSION NEXT YEAR</code></p><p><code>text_2: THE UNITED STATES UNDERTOOK TO DEFEND WESTERN EUROPE AGAINST SOVIET ATTACK</code></p><p>(This may take a while for the speech recognition.)</p><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span>import espnet_model_zoo</span></span>
<span class="line"><span>from espnet_model_zoo.downloader import ModelDownloader</span></span>
<span class="line"><span>from espnet2.bin.asr_inference import Speech2Text</span></span>
<span class="line"><span></span></span>
<span class="line"><span>wsj_8k_model_url=&quot;https://zenodo.org/record/4012264/files/asr_train_asr_transformer_raw_char_1gpu_valid.acc.ave.zip?download=1&quot;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>d = ModelDownloader()</span></span>
<span class="line"><span>speech2text = Speech2Text(</span></span>
<span class="line"><span>  **d.download_and_unpack(wsj_8k_model_url),</span></span>
<span class="line"><span>  device=&quot;cuda:0&quot;,</span></span>
<span class="line"><span>)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>text_est = [None, None]</span></span>
<span class="line"><span>text_est[0], *_ = speech2text(waves_wsj[0].squeeze())[0]</span></span>
<span class="line"><span>text_est[1], *_ = speech2text(waves_wsj[1].squeeze())[0]</span></span>
<span class="line"><span>text_m, *_ = speech2text(mixwav)[0]</span></span>
<span class="line"><span>print(&quot;Mix Speech to Text: &quot;, text_m)</span></span>
<span class="line"><span>print(&quot;Separated Speech 1 to Text: &quot;, text_est[0])</span></span>
<span class="line"><span>print(&quot;Separated Speech 2 to Text: &quot;, text_est[1])</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span>import difflib</span></span>
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
<span class="line"><span>  print(&quot;Edit Distance = {}\\n&quot;.format(string_edit[perm_idx][i]), flush=True)</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>`,43),p=[l];function d(t,r){return a(),n("div",null,p)}const u=s(i,[["render",d],["__file","se_demo.html.vue"]]),v=JSON.parse('{"path":"/notebook/ESPnet2/Demo/SE/se_demo.html","title":"ESPnet Speech Enhancement Demonstration","lang":"en-US","frontmatter":{},"headers":[{"level":2,"title":"Install","slug":"install","link":"#install","children":[]},{"level":2,"title":"Speech Enhancement","slug":"speech-enhancement","link":"#speech-enhancement","children":[{"level":3,"title":"Single-Channel Enhancement, the CHiME example","slug":"single-channel-enhancement-the-chime-example","link":"#single-channel-enhancement-the-chime-example","children":[]},{"level":3,"title":"Enhance your own pre-recordings","slug":"enhance-your-own-pre-recordings","link":"#enhance-your-own-pre-recordings","children":[]},{"level":3,"title":"Multi-Channel Enhancement","slug":"multi-channel-enhancement","link":"#multi-channel-enhancement","children":[]}]},{"level":2,"title":"Speech Separation","slug":"speech-separation","link":"#speech-separation","children":[{"level":3,"title":"Model Selection","slug":"model-selection","link":"#model-selection","children":[]},{"level":3,"title":"Separate Speech Mixture","slug":"separate-speech-mixture","link":"#separate-speech-mixture","children":[]}]},{"level":2,"title":"Evluate separated speech with pretrained ASR model","slug":"evluate-separated-speech-with-pretrained-asr-model","link":"#evluate-separated-speech-with-pretrained-asr-model","children":[]}],"git":{"createdTime":null,"updatedTime":null,"contributors":[]},"readingTime":{"minutes":4.73,"words":1418},"filePathRelative":"notebook/ESPnet2/Demo/SE/se_demo.md","excerpt":"\\n<figure><a href=\\"https://colab.research.google.com/drive/1fjRJCh96SoYLZPRxsjF9VDv4Q2VoIckI?usp=sharing\\" target=\\"_blank\\" rel=\\"noopener noreferrer\\"><img src=\\"https://colab.research.google.com/assets/colab-badge.svg\\" alt=\\"Open In Colab\\" tabindex=\\"0\\" loading=\\"lazy\\"></a><figcaption>Open In Colab</figcaption></figure>"}');export{u as comp,v as data};
