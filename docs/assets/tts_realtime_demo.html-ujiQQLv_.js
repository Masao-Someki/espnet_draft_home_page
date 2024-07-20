import{_ as s}from"./plugin-vue_export-helper-DlAUqK2U.js";import{c as n,o as a,a as e}from"./app-KOUU_Wij.js";const i={},l=e(`<figure><a href="https://colab.research.google.com/github/espnet/notebook/blob/master/tts_realtime_demo.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" tabindex="0" loading="lazy"></a><figcaption>Open In Colab</figcaption></figure><h1 id="espnet-real-time-e2e-tts-demonstration" tabindex="-1"><a class="header-anchor" href="#espnet-real-time-e2e-tts-demonstration"><span>ESPnet real time E2E-TTS demonstration</span></a></h1><p>This notebook provides a demonstration of the realtime E2E-TTS using ESPnet-TTS and ParallelWaveGAN (+ MelGAN).</p><ul><li>ESPnet: https://github.com/espnet/espnet</li><li>ParallelWaveGAN: https://github.com/kan-bayashi/ParallelWaveGAN</li></ul><p>Author: Tomoki Hayashi (<a href="https://github.com/kan-bayashi" target="_blank" rel="noopener noreferrer">@kan-bayashi</a>)</p><h2 id="install" tabindex="-1"><a class="header-anchor" href="#install"><span>Install</span></a></h2><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span># install minimal components</span></span>
<span class="line"><span>!pip install -q parallel_wavegan PyYaml unidecode ConfigArgparse g2p_en espnet_tts_frontend</span></span>
<span class="line"><span>!pip install --upgrade --no-cache-dir gdown</span></span>
<span class="line"><span>!git clone -q https://github.com/espnet/espnet.git</span></span>
<span class="line"><span>!cd espnet &amp;&amp; git fetch &amp;&amp; git checkout -b v.0.9.1 refs/tags/v.0.9.1</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><hr><h2 id="english-demo" tabindex="-1"><a class="header-anchor" href="#english-demo"><span>English demo</span></a></h2><h3 id="download-pretrained-feature-generation-model" tabindex="-1"><a class="header-anchor" href="#download-pretrained-feature-generation-model"><span>Download pretrained feature generation model</span></a></h3><p>You can select one from three models. Please only run the seletected model cells.</p><h4 id="a-tacotron2" tabindex="-1"><a class="header-anchor" href="#a-tacotron2"><span>(a) Tacotron2</span></a></h4><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span># download pretrained model</span></span>
<span class="line"><span>import os</span></span>
<span class="line"><span>if not os.path.exists(&quot;downloads/en/tacotron2&quot;):</span></span>
<span class="line"><span>    !./espnet/utils/download_from_google_drive.sh \\</span></span>
<span class="line"><span>        https://drive.google.com/open?id=1lFfeyewyOsxaNO-DEWy9iSz6qB9ZS1UR downloads/en/tacotron2 tar.gz</span></span>
<span class="line"><span></span></span>
<span class="line"><span># set path</span></span>
<span class="line"><span>trans_type = &quot;phn&quot;</span></span>
<span class="line"><span>dict_path = &quot;downloads/en/tacotron2/data/lang_1phn/phn_train_no_dev_units.txt&quot;</span></span>
<span class="line"><span>model_path = &quot;downloads/en/tacotron2/exp/phn_train_no_dev_pytorch_train_pytorch_tacotron2.v3/results/model.last1.avg.best&quot;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>print(&quot;sucessfully finished download.&quot;)</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h4 id="b-transformer" tabindex="-1"><a class="header-anchor" href="#b-transformer"><span>(b) Transformer</span></a></h4><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span># download pretrained model</span></span>
<span class="line"><span>import os</span></span>
<span class="line"><span>if not os.path.exists(&quot;downloads/en/transformer&quot;):</span></span>
<span class="line"><span>    !./espnet/utils/download_from_google_drive.sh \\</span></span>
<span class="line"><span>        https://drive.google.com/open?id=1z8KSOWVBjK-_Ws4RxVN4NTx-Buy03-7c downloads/en/transformer tar.gz</span></span>
<span class="line"><span></span></span>
<span class="line"><span># set path</span></span>
<span class="line"><span>trans_type = &quot;phn&quot;</span></span>
<span class="line"><span>dict_path = &quot;downloads/en/transformer/data/lang_1phn/phn_train_no_dev_units.txt&quot;</span></span>
<span class="line"><span>model_path = &quot;downloads/en/transformer/exp/phn_train_no_dev_pytorch_train_pytorch_transformer.v3.single/results/model.last1.avg.best&quot;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>print(&quot;sucessfully finished download.&quot;)</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h4 id="c-fastspeech" tabindex="-1"><a class="header-anchor" href="#c-fastspeech"><span>(c) FastSpeech</span></a></h4><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span># download pretrained model</span></span>
<span class="line"><span>import os</span></span>
<span class="line"><span>if not os.path.exists(&quot;downloads/en/fastspeech&quot;):</span></span>
<span class="line"><span>    !./espnet/utils/download_from_google_drive.sh \\</span></span>
<span class="line"><span>        https://drive.google.com/open?id=1P9I4qag8wAcJiTCPawt6WCKBqUfJFtFp downloads/en/fastspeech tar.gz</span></span>
<span class="line"><span></span></span>
<span class="line"><span># set path</span></span>
<span class="line"><span>trans_type = &quot;phn&quot;</span></span>
<span class="line"><span>dict_path = &quot;downloads/en/fastspeech/data/lang_1phn/phn_train_no_dev_units.txt&quot;</span></span>
<span class="line"><span>model_path = &quot;downloads/en/fastspeech/exp/phn_train_no_dev_pytorch_train_tacotron2.v3_fastspeech.v4.single/results/model.last1.avg.best&quot;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>print(&quot;Sucessfully finished download.&quot;)</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="download-pretrained-vocoder-model" tabindex="-1"><a class="header-anchor" href="#download-pretrained-vocoder-model"><span>Download pretrained vocoder model</span></a></h3><p>You can select one from two models. Please only run the seletected model cells.</p><h4 id="a-parallel-wavegan" tabindex="-1"><a class="header-anchor" href="#a-parallel-wavegan"><span>(a) Parallel WaveGAN</span></a></h4><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span># download pretrained model</span></span>
<span class="line"><span>import os</span></span>
<span class="line"><span>if not os.path.exists(&quot;downloads/en/parallel_wavegan&quot;):</span></span>
<span class="line"><span>    !./espnet/utils/download_from_google_drive.sh \\</span></span>
<span class="line"><span>        https://drive.google.com/open?id=1Grn7X9wD35UcDJ5F7chwdTqTa4U7DeVB downloads/en/parallel_wavegan tar.gz</span></span>
<span class="line"><span></span></span>
<span class="line"><span># set path</span></span>
<span class="line"><span>vocoder_path = &quot;downloads/en/parallel_wavegan/ljspeech.parallel_wavegan.v2/checkpoint-400000steps.pkl&quot;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>print(&quot;Sucessfully finished download.&quot;)</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h4 id="b-melgan" tabindex="-1"><a class="header-anchor" href="#b-melgan"><span>(b) MelGAN</span></a></h4><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span># download pretrained model</span></span>
<span class="line"><span>import os</span></span>
<span class="line"><span>if not os.path.exists(&quot;downloads/en/melgan&quot;):</span></span>
<span class="line"><span>    !./espnet/utils/download_from_google_drive.sh \\</span></span>
<span class="line"><span>        https://drive.google.com/open?id=1_a8faVA5OGCzIcJNw4blQYjfG4oA9VEt downloads/en/melgan tar.gz</span></span>
<span class="line"><span></span></span>
<span class="line"><span># set path</span></span>
<span class="line"><span>vocoder_path = &quot;downloads/en/melgan/train_nodev_ljspeech_melgan.v3.long/checkpoint-4000000steps.pkl&quot;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>print(&quot;Sucessfully finished download.&quot;)</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h4 id="c-multi-band-melgan" tabindex="-1"><a class="header-anchor" href="#c-multi-band-melgan"><span>(c) Multi-band MelGAN</span></a></h4><p>This is an <strong>EXPERIMENTAL</strong> model.</p><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span># download pretrained model</span></span>
<span class="line"><span>import os</span></span>
<span class="line"><span>if not os.path.exists(&quot;downloads/en/mb-melgan&quot;):</span></span>
<span class="line"><span>    !./espnet/utils/download_from_google_drive.sh \\</span></span>
<span class="line"><span>        https://drive.google.com/open?id=1rGG5y15uy4WZ-lJy8NPVTkmB_6VhC20V downloads/en/mb-melgan tar.gz</span></span>
<span class="line"><span></span></span>
<span class="line"><span># set path</span></span>
<span class="line"><span>vocoder_path = &quot;downloads/en/mb-melgan/train_nodev_ljspeech_multi_band_melgan.v1/checkpoint-1000000steps.pkl&quot;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>print(&quot;Sucessfully finished download.&quot;)</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="setup" tabindex="-1"><a class="header-anchor" href="#setup"><span>Setup</span></a></h3><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span># add path</span></span>
<span class="line"><span>import sys</span></span>
<span class="line"><span>sys.path.append(&quot;espnet&quot;)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># define device</span></span>
<span class="line"><span>import torch</span></span>
<span class="line"><span>device = torch.device(&quot;cuda&quot;)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># define E2E-TTS model</span></span>
<span class="line"><span>from argparse import Namespace</span></span>
<span class="line"><span>from espnet.asr.asr_utils import get_model_conf</span></span>
<span class="line"><span>from espnet.asr.asr_utils import torch_load</span></span>
<span class="line"><span>from espnet.utils.dynamic_import import dynamic_import</span></span>
<span class="line"><span>idim, odim, train_args = get_model_conf(model_path)</span></span>
<span class="line"><span>model_class = dynamic_import(train_args.model_module)</span></span>
<span class="line"><span>model = model_class(idim, odim, train_args)</span></span>
<span class="line"><span>torch_load(model_path, model)</span></span>
<span class="line"><span>model = model.eval().to(device)</span></span>
<span class="line"><span>inference_args = Namespace(**{</span></span>
<span class="line"><span>    &quot;threshold&quot;: 0.5,&quot;minlenratio&quot;: 0.0, &quot;maxlenratio&quot;: 10.0,</span></span>
<span class="line"><span>    # Only for Tacotron 2</span></span>
<span class="line"><span>    &quot;use_attention_constraint&quot;: True, &quot;backward_window&quot;: 1,&quot;forward_window&quot;:3,</span></span>
<span class="line"><span>    # Only for fastspeech (lower than 1.0 is faster speech, higher than 1.0 is slower speech)</span></span>
<span class="line"><span>    &quot;fastspeech_alpha&quot;: 1.0,</span></span>
<span class="line"><span>    })</span></span>
<span class="line"><span></span></span>
<span class="line"><span># define neural vocoder</span></span>
<span class="line"><span>from parallel_wavegan.utils import load_model</span></span>
<span class="line"><span>fs = 22050</span></span>
<span class="line"><span>vocoder = load_model(vocoder_path)</span></span>
<span class="line"><span>vocoder.remove_weight_norm()</span></span>
<span class="line"><span>vocoder = vocoder.eval().to(device)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># define text frontend</span></span>
<span class="line"><span>from tacotron_cleaner.cleaners import custom_english_cleaners</span></span>
<span class="line"><span>from g2p_en import G2p</span></span>
<span class="line"><span>with open(dict_path) as f:</span></span>
<span class="line"><span>    lines = f.readlines()</span></span>
<span class="line"><span>lines = [line.replace(&quot;\\n&quot;, &quot;&quot;).split(&quot; &quot;) for line in lines]</span></span>
<span class="line"><span>char_to_id = {c: int(i) for c, i in lines}</span></span>
<span class="line"><span>g2p = G2p()</span></span>
<span class="line"><span>def frontend(text):</span></span>
<span class="line"><span>    &quot;&quot;&quot;Clean text and then convert to id sequence.&quot;&quot;&quot;</span></span>
<span class="line"><span>    text = custom_english_cleaners(text)</span></span>
<span class="line"><span>    </span></span>
<span class="line"><span>    if trans_type == &quot;phn&quot;:</span></span>
<span class="line"><span>        text = filter(lambda s: s != &quot; &quot;, g2p(text))</span></span>
<span class="line"><span>        text = &quot; &quot;.join(text)</span></span>
<span class="line"><span>        print(f&quot;Cleaned text: {text}&quot;)</span></span>
<span class="line"><span>        charseq = text.split(&quot; &quot;)</span></span>
<span class="line"><span>    else:</span></span>
<span class="line"><span>        print(f&quot;Cleaned text: {text}&quot;)</span></span>
<span class="line"><span>        charseq = list(text)</span></span>
<span class="line"><span>    idseq = []</span></span>
<span class="line"><span>    for c in charseq:</span></span>
<span class="line"><span>        if c.isspace():</span></span>
<span class="line"><span>            idseq += [char_to_id[&#39;&amp;lt;space&amp;gt;&#39;]]</span></span>
<span class="line"><span>        elif c not in char_to_id.keys():</span></span>
<span class="line"><span>            idseq += [char_to_id[&#39;&amp;lt;unk&amp;gt;&#39;]]</span></span>
<span class="line"><span>        else:</span></span>
<span class="line"><span>            idseq += [char_to_id[c]]</span></span>
<span class="line"><span>    idseq += [idim - 1]  # &amp;lt;eos&amp;gt;</span></span>
<span class="line"><span>    return torch.LongTensor(idseq).view(-1).to(device)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>import nltk</span></span>
<span class="line"><span>nltk.download(&#39;punkt&#39;)</span></span>
<span class="line"><span>print(&quot;Now ready to synthesize!&quot;)</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="synthesis" tabindex="-1"><a class="header-anchor" href="#synthesis"><span>Synthesis</span></a></h3><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span>import time</span></span>
<span class="line"><span>print(&quot;Input your favorite sentence in English!&quot;)</span></span>
<span class="line"><span>input_text = input()</span></span>
<span class="line"><span>with torch.no_grad():</span></span>
<span class="line"><span>    start = time.time()</span></span>
<span class="line"><span>    x = frontend(input_text)</span></span>
<span class="line"><span>    c, _, _ = model.inference(x, inference_args)</span></span>
<span class="line"><span>    y = vocoder.inference(c)</span></span>
<span class="line"><span>rtf = (time.time() - start) / (len(y) / fs)</span></span>
<span class="line"><span>print(f&quot;RTF = {rtf:5f}&quot;)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>from IPython.display import display, Audio</span></span>
<span class="line"><span>display(Audio(y.view(-1).cpu().numpy(), rate=fs))</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><hr><h2 id="japanese-demo" tabindex="-1"><a class="header-anchor" href="#japanese-demo"><span>Japanese demo</span></a></h2><h3 id="install-japanese-dependencies" tabindex="-1"><a class="header-anchor" href="#install-japanese-dependencies"><span>Install Japanese dependencies</span></a></h3><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span>!pip install pyopenjtalk</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div></div></div><h3 id="download-pretrained-models" tabindex="-1"><a class="header-anchor" href="#download-pretrained-models"><span>Download pretrained models</span></a></h3><p>Here we select Tacotron2 or Transformer. The vocoder model is Parallel WaveGAN.</p><h4 id="a-tacotron-2" tabindex="-1"><a class="header-anchor" href="#a-tacotron-2"><span>(a) Tacotron 2</span></a></h4><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span># download pretrained models</span></span>
<span class="line"><span>import os</span></span>
<span class="line"><span>if not os.path.exists(&quot;downloads/jp/tacotron2&quot;):</span></span>
<span class="line"><span>    !./espnet/utils/download_from_google_drive.sh \\</span></span>
<span class="line"><span>        https://drive.google.com/open?id=1OwrUQzAmvjj1x9cDhnZPp6dqtsEqGEJM downloads/jp/tacotron2 tar.gz</span></span>
<span class="line"><span>    !./espnet/utils/download_from_google_drive.sh \\</span></span>
<span class="line"><span>        https://drive.google.com/open?id=1kp5M4VvmagDmYckFJa78WGqh1drb_P9t downloads/jp/tacotron2 tar.gz</span></span>
<span class="line"><span></span></span>
<span class="line"><span># set path</span></span>
<span class="line"><span>dict_path = &quot;downloads/jp/tacotron2/data/lang_1phn/train_no_dev_units.txt&quot;</span></span>
<span class="line"><span>model_path = &quot;downloads/jp/tacotron2/exp/train_no_dev_pytorch_train_pytorch_tacotron2_phn/results/model.last1.avg.best&quot;</span></span>
<span class="line"><span>vocoder_path = &quot;downloads/jp/tacotron2/jsut.parallel_wavegan.v1/checkpoint-400000steps.pkl&quot;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>print(&quot;sucessfully finished download.&quot;)</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h4 id="b-transformer-1" tabindex="-1"><a class="header-anchor" href="#b-transformer-1"><span>(b) Transformer</span></a></h4><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span># download pretrained models</span></span>
<span class="line"><span>import os</span></span>
<span class="line"><span>if not os.path.exists(&quot;downloads/jp/transformer&quot;):</span></span>
<span class="line"><span>    !./espnet/utils/download_from_google_drive.sh \\</span></span>
<span class="line"><span>        https://drive.google.com/open?id=1OwrUQzAmvjj1x9cDhnZPp6dqtsEqGEJM downloads/jp/transformer tar.gz</span></span>
<span class="line"><span>    !./espnet/utils/download_from_google_drive.sh \\</span></span>
<span class="line"><span>        https://drive.google.com/open?id=1mEnZfBKqA4eT6Bn0eRZuP6lNzL-IL3VD downloads/jp/transformer tar.gz</span></span>
<span class="line"><span></span></span>
<span class="line"><span># set path</span></span>
<span class="line"><span>dict_path = &quot;downloads/jp/transformer/data/lang_1phn/train_no_dev_units.txt&quot;</span></span>
<span class="line"><span>model_path = &quot;downloads/jp/transformer/exp/train_no_dev_pytorch_train_pytorch_transformer_phn/results/model.last1.avg.best&quot;</span></span>
<span class="line"><span>vocoder_path = &quot;downloads/jp/transformer/jsut.parallel_wavegan.v1/checkpoint-400000steps.pkl&quot;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>print(&quot;sucessfully finished download.&quot;)</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="setup-1" tabindex="-1"><a class="header-anchor" href="#setup-1"><span>Setup</span></a></h3><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span># add path</span></span>
<span class="line"><span>import sys</span></span>
<span class="line"><span>sys.path.append(&quot;espnet&quot;)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># define device</span></span>
<span class="line"><span>import torch</span></span>
<span class="line"><span>device = torch.device(&quot;cuda&quot;)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># define E2E-TTS model</span></span>
<span class="line"><span>from argparse import Namespace</span></span>
<span class="line"><span>from espnet.asr.asr_utils import get_model_conf</span></span>
<span class="line"><span>from espnet.asr.asr_utils import torch_load</span></span>
<span class="line"><span>from espnet.utils.dynamic_import import dynamic_import</span></span>
<span class="line"><span>idim, odim, train_args = get_model_conf(model_path)</span></span>
<span class="line"><span>model_class = dynamic_import(train_args.model_module)</span></span>
<span class="line"><span>model = model_class(idim, odim, train_args)</span></span>
<span class="line"><span>torch_load(model_path, model)</span></span>
<span class="line"><span>model = model.eval().to(device)</span></span>
<span class="line"><span>inference_args = Namespace(**{&quot;threshold&quot;: 0.5, &quot;minlenratio&quot;: 0.0, &quot;maxlenratio&quot;: 10.0})</span></span>
<span class="line"><span></span></span>
<span class="line"><span># define neural vocoder</span></span>
<span class="line"><span>from parallel_wavegan.utils import load_model</span></span>
<span class="line"><span>fs = 24000</span></span>
<span class="line"><span>vocoder = load_model(vocoder_path)</span></span>
<span class="line"><span>vocoder.remove_weight_norm()</span></span>
<span class="line"><span>vocoder = vocoder.eval().to(device)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># define text frontend</span></span>
<span class="line"><span>import pyopenjtalk</span></span>
<span class="line"><span>with open(dict_path) as f:</span></span>
<span class="line"><span>    lines = f.readlines()</span></span>
<span class="line"><span>lines = [line.replace(&quot;\\n&quot;, &quot;&quot;).split(&quot; &quot;) for line in lines]</span></span>
<span class="line"><span>char_to_id = {c: int(i) for c, i in lines}</span></span>
<span class="line"><span>def frontend(text):</span></span>
<span class="line"><span>    &quot;&quot;&quot;Clean text and then convert to id sequence.&quot;&quot;&quot;</span></span>
<span class="line"><span>    text = pyopenjtalk.g2p(text, kana=False)</span></span>
<span class="line"><span>    print(f&quot;Cleaned text: {text}&quot;)</span></span>
<span class="line"><span>    charseq = text.split(&quot; &quot;)</span></span>
<span class="line"><span>    idseq = []</span></span>
<span class="line"><span>    for c in charseq:</span></span>
<span class="line"><span>        if c.isspace():</span></span>
<span class="line"><span>            idseq += [char_to_id[&#39;&amp;lt;space&amp;gt;&#39;]]</span></span>
<span class="line"><span>        elif c not in char_to_id.keys():</span></span>
<span class="line"><span>            idseq += [char_to_id[&#39;&amp;lt;unk&amp;gt;&#39;]]</span></span>
<span class="line"><span>        else:</span></span>
<span class="line"><span>            idseq += [char_to_id[c]]</span></span>
<span class="line"><span>    idseq += [idim - 1]  # &amp;lt;eos&amp;gt;</span></span>
<span class="line"><span>    return torch.LongTensor(idseq).view(-1).to(device)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>frontend(&quot;初回の辞書のインストールが必要です&quot;)</span></span>
<span class="line"><span>print(&quot;Now ready to synthesize!&quot;)</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="synthesis-1" tabindex="-1"><a class="header-anchor" href="#synthesis-1"><span>Synthesis</span></a></h3><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span>import time</span></span>
<span class="line"><span>print(&quot;日本語で好きな文章を入力してください&quot;)</span></span>
<span class="line"><span>input_text = input()</span></span>
<span class="line"><span></span></span>
<span class="line"><span>with torch.no_grad():</span></span>
<span class="line"><span>    start = time.time()</span></span>
<span class="line"><span>    x = frontend(input_text)</span></span>
<span class="line"><span>    c, _, _ = model.inference(x, inference_args)</span></span>
<span class="line"><span>    y = vocoder.inference(c)</span></span>
<span class="line"><span>rtf = (time.time() - start) / (len(y) / fs)</span></span>
<span class="line"><span>print(f&quot;RTF = {rtf:5f}&quot;)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>from IPython.display import display, Audio</span></span>
<span class="line"><span>display(Audio(y.view(-1).cpu().numpy(), rate=fs))</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><hr><h2 id="mandarin-demo" tabindex="-1"><a class="header-anchor" href="#mandarin-demo"><span>Mandarin demo</span></a></h2><p><strong>IMPORTANT NOTE</strong>: The author cannot understand Mandarin. The text front-end part might have some bugs.</p><h3 id="install-mandarin-dependencies" tabindex="-1"><a class="header-anchor" href="#install-mandarin-dependencies"><span>Install Mandarin dependencies</span></a></h3><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span>!pip install pypinyin</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div></div></div><h3 id="download-pretrained-models-1" tabindex="-1"><a class="header-anchor" href="#download-pretrained-models-1"><span>Download pretrained models</span></a></h3><p>You can select Transformer or FastSpeech.</p><h4 id="a-transformer" tabindex="-1"><a class="header-anchor" href="#a-transformer"><span>(a) Transformer</span></a></h4><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span># download pretrained models</span></span>
<span class="line"><span>import os</span></span>
<span class="line"><span>if not os.path.exists(&quot;downloads/zh/transformer&quot;):</span></span>
<span class="line"><span>    !./espnet/utils/download_from_google_drive.sh \\</span></span>
<span class="line"><span>        https://drive.google.com/open?id=10M6H88jEUGbRWBmU1Ff2VaTmOAeL8CEy downloads/zh/transformer tar.gz</span></span>
<span class="line"><span>    !./espnet/utils/download_from_google_drive.sh \\</span></span>
<span class="line"><span>        https://drive.google.com/open?id=1bTSygvonv5TS6-iuYsOIUWpN2atGnyhZ downloads/zh/transformer tar.gz</span></span>
<span class="line"><span></span></span>
<span class="line"><span># set path</span></span>
<span class="line"><span>dict_path = &quot;downloads/zh/transformer/data/lang_phn/train_no_dev_units.txt&quot;</span></span>
<span class="line"><span>model_path = &quot;downloads/zh/transformer/exp/train_no_dev_pytorch_train_pytorch_transformer.v1.single/results/model.last1.avg.best&quot;</span></span>
<span class="line"><span>vocoder_path = &quot;downloads/zh/transformer/csmsc.parallel_wavegan.v1/checkpoint-400000steps.pkl&quot;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>print(&quot;sucessfully finished download.&quot;)</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h4 id="b-fastspeech" tabindex="-1"><a class="header-anchor" href="#b-fastspeech"><span>(b) FastSpeech</span></a></h4><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span># download pretrained models</span></span>
<span class="line"><span>import os</span></span>
<span class="line"><span>if not os.path.exists(&quot;downloads/zh/fastspeech&quot;):</span></span>
<span class="line"><span>    !./espnet/utils/download_from_google_drive.sh \\</span></span>
<span class="line"><span>        https://drive.google.com/open?id=10M6H88jEUGbRWBmU1Ff2VaTmOAeL8CEy downloads/zh/fastspeech tar.gz</span></span>
<span class="line"><span>    !./espnet/utils/download_from_google_drive.sh \\</span></span>
<span class="line"><span>        https://drive.google.com/open?id=1T8thxkAxjGFPXPWPTcKLvHnd6lG0-82R downloads/zh/fastspeech tar.gz </span></span>
<span class="line"><span></span></span>
<span class="line"><span># set path</span></span>
<span class="line"><span>dict_path = &quot;downloads/zh/fastspeech/data/lang_phn/train_no_dev_units.txt&quot;</span></span>
<span class="line"><span>model_path = &quot;downloads/zh/fastspeech/exp/train_no_dev_pytorch_train_fastspeech.v3.single/results/model.last1.avg.best&quot;</span></span>
<span class="line"><span>vocoder_path = &quot;downloads/zh/fastspeech/csmsc.parallel_wavegan.v1/checkpoint-400000steps.pkl&quot;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>print(&quot;sucessfully finished download.&quot;)</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="setup-2" tabindex="-1"><a class="header-anchor" href="#setup-2"><span>Setup</span></a></h3><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span># add path</span></span>
<span class="line"><span>import sys</span></span>
<span class="line"><span>sys.path.append(&quot;espnet&quot;)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># define device</span></span>
<span class="line"><span>import torch</span></span>
<span class="line"><span>device = torch.device(&quot;cuda&quot;)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># define E2E-TTS model</span></span>
<span class="line"><span>from argparse import Namespace</span></span>
<span class="line"><span>from espnet.asr.asr_utils import get_model_conf</span></span>
<span class="line"><span>from espnet.asr.asr_utils import torch_load</span></span>
<span class="line"><span>from espnet.utils.dynamic_import import dynamic_import</span></span>
<span class="line"><span>idim, odim, train_args = get_model_conf(model_path)</span></span>
<span class="line"><span>model_class = dynamic_import(train_args.model_module)</span></span>
<span class="line"><span>model = model_class(idim, odim, train_args)</span></span>
<span class="line"><span>torch_load(model_path, model)</span></span>
<span class="line"><span>model = model.eval().to(device)</span></span>
<span class="line"><span>inference_args = Namespace(**{&quot;threshold&quot;: 0.5, &quot;minlenratio&quot;: 0.0, &quot;maxlenratio&quot;: 10.0})</span></span>
<span class="line"><span></span></span>
<span class="line"><span># define neural vocoder</span></span>
<span class="line"><span>from parallel_wavegan.utils import load_model</span></span>
<span class="line"><span>fs = 24000</span></span>
<span class="line"><span>vocoder = load_model(vocoder_path)</span></span>
<span class="line"><span>vocoder.remove_weight_norm()</span></span>
<span class="line"><span>vocoder = vocoder.eval().to(device)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># define text frontend</span></span>
<span class="line"><span>from pypinyin import pinyin, Style</span></span>
<span class="line"><span>from pypinyin.style._utils import get_initials, get_finals</span></span>
<span class="line"><span>with open(dict_path) as f:</span></span>
<span class="line"><span>    lines = f.readlines()</span></span>
<span class="line"><span>lines = [line.replace(&quot;\\n&quot;, &quot;&quot;).split(&quot; &quot;) for line in lines]</span></span>
<span class="line"><span>char_to_id = {c: int(i) for c, i in lines}</span></span>
<span class="line"><span>def frontend(text):</span></span>
<span class="line"><span>    &quot;&quot;&quot;Clean text and then convert to id sequence.&quot;&quot;&quot;</span></span>
<span class="line"><span>    text = pinyin(text, style=Style.TONE3)</span></span>
<span class="line"><span>    text = [c[0] for c in text]</span></span>
<span class="line"><span>    print(f&quot;Cleaned text: {text}&quot;)</span></span>
<span class="line"><span>    idseq = []</span></span>
<span class="line"><span>    for x in text:</span></span>
<span class="line"><span>        c_init = get_initials(x, strict=True)</span></span>
<span class="line"><span>        c_final = get_finals(x, strict=True)</span></span>
<span class="line"><span>        for c in [c_init, c_final]:</span></span>
<span class="line"><span>            if len(c) == 0:</span></span>
<span class="line"><span>                continue</span></span>
<span class="line"><span>            c = c.replace(&quot;ü&quot;, &quot;v&quot;)</span></span>
<span class="line"><span>            c = c.replace(&quot;ui&quot;, &quot;uei&quot;)</span></span>
<span class="line"><span>            c = c.replace(&quot;un&quot;, &quot;uen&quot;)</span></span>
<span class="line"><span>            c = c.replace(&quot;iu&quot;, &quot;iou&quot;)</span></span>
<span class="line"><span>            # Special rule: &quot;e5n&quot; -&gt; &quot;en5&quot;</span></span>
<span class="line"><span>            if &quot;5&quot; in c:</span></span>
<span class="line"><span>                c = c.replace(&quot;5&quot;, &quot;&quot;) + &quot;5&quot;</span></span>
<span class="line"><span>            if c not in char_to_id.keys():</span></span>
<span class="line"><span>                print(f&quot;WARN: {c} is not included in dict.&quot;)</span></span>
<span class="line"><span>                idseq += [char_to_id[&#39;&amp;lt;unk&amp;gt;&#39;]]</span></span>
<span class="line"><span>            else:</span></span>
<span class="line"><span>                idseq += [char_to_id[c]]</span></span>
<span class="line"><span>    idseq += [idim - 1]  # &amp;lt;eos&amp;gt;</span></span>
<span class="line"><span>    return torch.LongTensor(idseq).view(-1).to(device)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>print(&quot;now ready to synthesize!&quot;)</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="synthesis-2" tabindex="-1"><a class="header-anchor" href="#synthesis-2"><span>Synthesis</span></a></h3><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" data-title="" style="--shiki-light:#24292e;--shiki-dark:#abb2bf;--shiki-light-bg:#fff;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes github-light one-dark-pro vp-code"><code><span class="line"><span>import time</span></span>
<span class="line"><span>print(&quot;請用中文輸入您喜歡的句子!&quot;)</span></span>
<span class="line"><span>input_text = input()</span></span>
<span class="line"><span></span></span>
<span class="line"><span>with torch.no_grad():</span></span>
<span class="line"><span>    start = time.time()</span></span>
<span class="line"><span>    x = frontend(input_text)</span></span>
<span class="line"><span>    c, _, _ = model.inference(x, inference_args)</span></span>
<span class="line"><span>    y = vocoder.inference(c)</span></span>
<span class="line"><span>rtf = (time.time() - start) / (len(y) / fs)</span></span>
<span class="line"><span>print(f&quot;RTF = {rtf:5f}&quot;)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>from IPython.display import display, Audio</span></span>
<span class="line"><span>display(Audio(y.view(-1).cpu().numpy(), rate=fs))</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>`,59),p=[l];function d(t,r){return a(),n("div",null,p)}const v=s(i,[["render",d],["__file","tts_realtime_demo.html.vue"]]),u=JSON.parse('{"path":"/notebook/ESPnet1/tts_realtime_demo.html","title":"ESPnet real time E2E-TTS demonstration","lang":"en-US","frontmatter":{},"headers":[{"level":2,"title":"Install","slug":"install","link":"#install","children":[]},{"level":2,"title":"English demo","slug":"english-demo","link":"#english-demo","children":[{"level":3,"title":"Download pretrained feature generation model","slug":"download-pretrained-feature-generation-model","link":"#download-pretrained-feature-generation-model","children":[]},{"level":3,"title":"Download pretrained vocoder model","slug":"download-pretrained-vocoder-model","link":"#download-pretrained-vocoder-model","children":[]},{"level":3,"title":"Setup","slug":"setup","link":"#setup","children":[]},{"level":3,"title":"Synthesis","slug":"synthesis","link":"#synthesis","children":[]}]},{"level":2,"title":"Japanese demo","slug":"japanese-demo","link":"#japanese-demo","children":[{"level":3,"title":"Install Japanese dependencies","slug":"install-japanese-dependencies","link":"#install-japanese-dependencies","children":[]},{"level":3,"title":"Download pretrained models","slug":"download-pretrained-models","link":"#download-pretrained-models","children":[]},{"level":3,"title":"Setup","slug":"setup-1","link":"#setup-1","children":[]},{"level":3,"title":"Synthesis","slug":"synthesis-1","link":"#synthesis-1","children":[]}]},{"level":2,"title":"Mandarin demo","slug":"mandarin-demo","link":"#mandarin-demo","children":[{"level":3,"title":"Install Mandarin dependencies","slug":"install-mandarin-dependencies","link":"#install-mandarin-dependencies","children":[]},{"level":3,"title":"Download pretrained models","slug":"download-pretrained-models-1","link":"#download-pretrained-models-1","children":[]},{"level":3,"title":"Setup","slug":"setup-2","link":"#setup-2","children":[]},{"level":3,"title":"Synthesis","slug":"synthesis-2","link":"#synthesis-2","children":[]}]}],"git":{"createdTime":null,"updatedTime":null,"contributors":[]},"readingTime":{"minutes":4.47,"words":1342},"filePathRelative":"notebook/ESPnet1/tts_realtime_demo.md","excerpt":"<figure><a href=\\"https://colab.research.google.com/github/espnet/notebook/blob/master/tts_realtime_demo.ipynb\\" target=\\"_blank\\" rel=\\"noopener noreferrer\\"><img src=\\"https://colab.research.google.com/assets/colab-badge.svg\\" alt=\\"Open In Colab\\" tabindex=\\"0\\" loading=\\"lazy\\"></a><figcaption>Open In Colab</figcaption></figure>"}');export{v as comp,u as data};
