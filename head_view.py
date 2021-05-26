import json
import os
import uuid

from IPython.core.display import display, HTML, Javascript

import torch


def format_attention(attention):
    squeezed = []
    for layer_attention in attention:
        print(layer_attention.shape)
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        squeezed.append(layer_attention.squeeze(0))
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)

def format_special_chars(tokens):
    return [t.replace('Ġ', ' ').replace('▁', ' ').replace('</w>', '') for t in tokens]



def head_view(attention, tokens, sentence_b_start = None, prettify_tokens=True, layer=None, heads=None):
    """Render head view
        Args:
            attention: list of ``torch.FloatTensor``(one for each layer) of shape
                ``(batch_size(must be 1), num_heads, sequence_length, sequence_length)``
            tokens: list of tokens
            sentence_b_start: index of first wordpiece in sentence B if input text is sentence pair (optional)
            prettify_tokens: indicates whether to remove special characters in wordpieces, e.g. Ġ
            layer: index of layer to show in visualization when first loads. If non specified, defaults to layer 0.
            heads: indices of heads to show in visualization when first loads. If non specified, defaults to all.
    """

    # Generate unique div id to enable multiple visualizations in one notebook
    vis_id = 'bertviz-%s'%(uuid.uuid4().hex)

    if sentence_b_start is not None:
        vis_html = """      
            <div id='%s'>
                <span style="user-select:none">
                    Layer: <select id="layer"></select>
                    Attention: <select id="filter">
                      <option value="all">All</option>
                      <option value="aa">Sentence A -> Sentence A</option>
                      <option value="ab">Sentence A -> Sentence B</option>
                      <option value="ba">Sentence B -> Sentence A</option>
                      <option value="bb">Sentence B -> Sentence B</option>
                    </select>
                    </span>
                <div id='vis'></div>
            </div>
        """%(vis_id)
    else:
        vis_html = """
            <div id='%s'>
              <span style="user-select:none">
                Layer: <select id="layer"></select>
              </span>
              <div id='vis'></div> 
            </div>
        """%(vis_id)

#     if prettify_tokens:
#         tokens = format_special_chars(tokens)

    attn = format_attention(attention)
    attn_data = {
        'all': {
            'attn': attn.tolist(),
            'left_text': tokens,
            'right_text': tokens
        }
    }

    params = {
        'attention': attn_data,
        'default_filter': "all",
        'root_div_id': vis_id,
        'layer': layer,
        'heads': heads
    }
    attn_seq_len = len(attn_data['all']['attn'][0][0])
#     if attn_seq_len != len(tokens):
#         raise ValueError(f"Attention has {attn_seq_len} positions, while number of tokens is {len(tokens)}")

    # require.js must be imported for Colab or JupyterLab:
    display(HTML('<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js"></script>'))
    display(HTML(vis_html))
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    vis_js = open(os.path.join(__location__, 'head_view.js')).read().replace("PYTHON_PARAMS", json.dumps(params))
    display(Javascript(vis_js))