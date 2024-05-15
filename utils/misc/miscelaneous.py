import json
import numpy as np
import scipy.stats as st


def ainfo(tag, t):
    print('info:', tag, t.shape, (t.min()), (t.mean()), (t.max()))


# kind of like non-pretty json.dumps(dictionary)
def recursive_dict_print(dictionary):
    print("{", end="")
    for index, key in enumerate(dictionary):
        if isinstance(dictionary[key], dict):
            if index != 0:
                print(", ", end="")
            print(key, end=": ")
            recursive_dict_print(dictionary[key])
        else:
            if index != 0:
                print(", ", end="")
            print("{}: {}".format(key, dictionary[key]), end="")
    print("}", end="")


# prints a text based histogram where each line of text is a bin
def text_hist(tag, values, num_bins=20,
              chars_for_max_line=100, line_end_char='-', line_indicator='|'):
    hist, edges = np.histogram(values, num_bins)

    print('Line Histogram for ' + tag + '.')

    def print_chars(chars):
        if len(chars) > 0:
            print(chars, end='')

    bin_size = hist.max() / chars_for_max_line
    for num, count in enumerate(hist):
        print_chars(line_indicator)  # start line

        line_len = int(np.ceil(count / bin_size))
        # 0-9 character to indicate how close to next bin
        print_chars('9' * (line_len - 1))
        last_line_char = str(int(np.ceil(9 * (1 - (line_len * bin_size - count) / bin_size)))) if count != 0 else '0'
        print_chars(last_line_char)
        print_chars(line_end_char * (chars_for_max_line - (line_len if line_len > 0 else 1)))
        print_chars(line_indicator)  # end line

        print_chars(': {} values in [{}, {}]'.format(count, edges[num], edges[num + 1]))
        print('')  # new line


def float2str3(value):
    return float2str(value, decimals=3)


def float2str(value, decimals=6):
    if value is not float:
        value = float(value)
    format_str = "{:." + str(decimals)
    if abs(value) < 10 ** -decimals and value != 0:
        format_str += "E"
    else:
        format_str += "f"
    format_str += "}"
    return format_str.format(value)


def bool2str(b):
    return "T" if b else "F"


def dictionary_to_json_string(dictionary):
    return json.dumps(dictionary, indent=4)


def dict_list_new_or_append(dictionary, key, value):
    if key in dictionary:
        dictionary[key].append(value)
    else:
        dictionary[key] = [value]


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""
    # print('gkern offset', offset, -nsig*(1-offset), nsig*(1+offset))
    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    return kern1d/kern1d.sum()


def lerp(a, b, ratio):
    return a + (b - a) * ratio


# lerps over a list instead of just 2 values
def lerp_list(l, ratio=0.5):
    l = np.array(l)
    n = len(l)
    r = int(ratio * (n-1))
    rr = (ratio * (n-1)) - r
    lr = l[r:r+2]
    if len(lr) < 2:
        return lr.flatten()
    else:
        return lerp(lr[0], lr[1], rr).flatten()


# lerps over a list of values and a sliding gaussian weight
def lerp_list_gau(l, ratio=0.5, nsig=10):
    lnp = np.array(l)
    n = len(l)
    i = int((1-ratio)*(n-1))
    weights = gkern(kernlen=2*n-1, nsig=nsig)[i:i+n]
    if len(lnp.shape) > 1:
        weights = weights.reshape(-1, 1)
    weighted = lnp * weights / weights.sum()
    return weighted.sum(axis=0) if len(lnp.shape) > 1 else weighted.sum()


def split_list(list, num_splits, append_leftover_to_last=False):
    """
    distributes a list of tasks into N splits
    :param list:
    :param num_splits:
    :param append_leftover_to_last: add leftover tasks to last thread or distribute to all threads equally
    :return:
    """
    n = len(list)
    list_per_split = [[] for _ in range(num_splits)]
    num_per_split = int(len(list) / num_splits)
    leftover = n - num_per_split * num_splits
    for i in range(num_splits):
        starting_index = i * num_per_split
        for item in list[starting_index: starting_index + num_per_split]:
            list_per_split[i].append(item)
    starting_index = num_splits * num_per_split
    for i, item in enumerate(list[starting_index: starting_index + leftover]):
        if append_leftover_to_last:
            index = num_splits - 1
        else:
            index = i % num_splits
        list_per_split[index].append(item)
    return list_per_split


def split_filename_and_extension(file_name):
    name_split_by_dot = file_name.split('.')
    img_filename_no_ext = '.'.join(name_split_by_dot[:-1])
    img_extension = name_split_by_dot[-1]
    return img_filename_no_ext, img_extension
