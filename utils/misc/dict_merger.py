def merge_dicts(dict1, dict2):
    merged = dict1.copy()
    for key in dict2:
        if isinstance(dict2[key], dict) and key in merged:
            merged[key] = merge_dicts(merged[key], dict2[key])
        else:
            if key in merged:
                print("Warning: merge_dicts() overwriting keys [{}] with content {}, "
                      "replacing with content {}.".
                      format(key, merged[key], dict2[key]))
            merged[key] = dict2[key]
    return merged


def merge_dict_list(dict_list, local=False):
    merged = {}
    for _dict in dict_list:
        for key in _dict:
            if local and key in merged:
                for key2 in _dict[key]:
                    print("Warning: merge_dict_list() overwriting keys [{}][{}] with content {}, "
                          "replacing with content {}.".
                          format(key, key2, merged[key][key2], _dict[key][key2]))
                    merged[key][key2] = _dict[key][key2]
            else:
                merged[key] = _dict[key]
    return merged


def test():
    d1 = {"bla1": {"psnr": {"a": 1, "b": 2}},
          "bla2": {"tmqi": {"a": "old", "b": "old"}}}
    d2 = {"bla2": {"tmqi": {"a": "new", "b": "new"}},
          "bla3": {"tmqi": {"a": 6, "b": 7}}}
    m = merge_dicts(d1, d2)

    for key in m:
        print(key, m[key])

    exit()


if __name__=="__main__":
    # test()

    import pickle as pkl

    from pprint import pprint

    # inputs
    path = "footage/images/subjStudy"

    pkl_path1 = "{}/{}".format(path, "stats_everything.pkl")
    pkl_path2 = "{}/{}".format(path, "stats-msssim.pkl")
    merged_path = "{}/merged/{}".format(path, "stats_12_merged.pkl")

    with open(pkl_path1, "rb") as pkl_file1, \
            open(pkl_path2, "rb") as pkl_file2, \
            open(merged_path, 'wb') as merged_file:
        stats1 = pkl.load(pkl_file1)
        stats2 = pkl.load(pkl_file2)

        # pprint(stats1)
        # pprint(stats2)

        print('Writing to', merged_path)
        merged_dict = merge_dicts(stats1, stats2)

        # pprint(merged_dict)

        pkl.dump(merged_dict, merged_file)
