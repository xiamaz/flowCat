"""
Operations to convert pandas structures into latex elements.
"""
import pandas as pd


def texclean(raw: str) -> str:
    """Escape special signs in latex."""
    return raw.replace("_", r"\_")


def df_tabulate(data: pd.DataFrame, level: int = 0) -> dict:
    """Create deduplicated dataframe structure."""
    result = {}
    max_level = len(data.index.names)
    for name, gdata in data.groupby(level=level):
        if level < max_level-1:
            result[name] = df_tabulate(gdata, level+1)
        else:
            result[name] = [
                " & ".join([texclean(str(t)) for t in r]) + r" \\"
                for r in gdata.apply(
                    lambda r: list(r.values), axis=1
                )
            ]

    output = [
        r for k, v in result.items()
        for r in [texclean(k) + " & " + v[0]] + ["  & " + r for r in v[1:]]
    ]
    return output

def df_to_table(data: pd.DataFrame, spec: str = ""):
    """Create latex table fragment from pandas data with smart printing
    of multiindexes."""
    # create title
    index_depth = len(data.index.names)
    if not spec:
        spec = "l"*(index_depth+data.shape[1])
    outtext = [
        r"\begin{{tabular}}{{{}}}".format(spec),
        r"\toprule",
    ]

    if index_depth > 1:
        names = [
            texclean(n) for n in list(data.index.names) + list(data.columns)
        ]
        outtext += [
            r"\multicolumn{{{}}}{{c}}{{Index}} \\".format(index_depth),
            r"\cmidrule(r){{1-{}}}".format(index_depth),
            " & ".join(names) + r" \\",
            r"\midrule",
        ]

    # create data
    outtext += list(
        df_tabulate(data)
    )

    # create footer
    outtext += [
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(outtext)


def df_save_latex(data: pd.DataFrame, path: str, *args, **kwargs):
    table = df_to_table(data, *args, **kwargs)
    with open(path, "w") as tfile:
        tfile.write(table)
