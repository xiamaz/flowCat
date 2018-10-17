"""Manage configuration for different systems."""
import toml

from . import utils


def infer_sections(data):
    """Infer section names from first level in data name.
    Will return a set of section names.
    """
    first_level = [k.split("_", 1)[0] for k in data]
    names = set(first_level)
    return names


def strip_section(data, section):
    """Strip the section name from all keys in data."""
    return {
        k[len(section) + 1:] if k.startswith(f"{section}_") else k: v
        for k, v in data.items()
    }


def build_sections(data, sections):
    """Build a mapping to sections to dicts containing the subkeys."""
    mapping = {}
    for section in sections:
        section_dict = {k: v for k, v in data.items() if k.startswith(f"{section}_")}
        mapping[section] = strip_section(section_dict, section)
    return mapping


def list_a_eq_b(list_a, list_b):
    if not isinstance(list_b, list):
        return False
    return str(sorted(list_a)) == str(sorted(list_b))

def dict_a_in_b(dict_a, dict_b):
    for k, v in dict_a.items():
        vb = dict_b.get(k, None)
        if isinstance(v, dict):
            if not dict_a_in_b(v, vb):
                return False
        elif isinstance(v, list):
            if not list_a_eq_b(v, vb):
                return False
        else:
            if v != vb:
                return False
    return True


def compare_configurations(conf_a, conf_b, section="", method="left"):
    """Compare a section of the given configurations. Check that
    config a is the same as b in the given section.
    Args:
        conf_a: Config object.
        conf_b: Another config object.
        section: Section to be compared. If falsy, will use entire config.
        method: Comparison method.
            left - All keys in a must be same in b
            right - All keys in b must be same in a
            both - All keys from both must be same
    Return:
        Boolean whether equality check has been passed.
    """
    if section:
        section_a = conf_a[section]
        section_b = conf_b[section]
    else:
        if isinstance(conf_a, Configuration):
            section_a = conf_a.dict
            section_b = conf_b.dict
        else:
            section_a = conf_a
            section_b = conf_b

    if method == "left":
        return dict_a_in_b(section_a, section_b)
    elif method == "right":
        return dict_a_in_b(section_b, section_a)
    elif method == "both":
        return dict_a_in_b(section_a, section_b) and dict_a_in_b(section_b, section_a)
    else:
        raise TypeError(f"Unknown method {method}")


def to_int_naming(data, key, tag):
    """Convert dict keys from strings to ints. This operation is done in-place.
    Args:
        data: Dict containing key values as parameters.
        key: Key of the data to be modified.
        tag: Tag on the string that will be replaced.
    Returns:
        Modified sectiondata dict.
    """
    if key in data:
        data[key] = {int(k.replace(tag, "")): v for k, v in data[key].items()}
    return data


def to_string_naming(data, key, tag):
    """Convert dicts with int names to strings. Adding the given tag."""
    if key in data:
        data[key] = {f"{tag}{k}": v for k, v in data[key].items()}
    return data


class Configuration:
    """Basic configuration class, allowing underline access to sectioned data."""

    def __init__(self, data, section=""):
        """Initialize from input data."""
        self.section = section
        self._data = data

    @property
    def dict(self):
        return {
            f"{self.section}_{section}_{k}": v
            for section, secdict in self._data.items() for k, v in secdict.items()
        }

    @classmethod
    def from_json(cls, path):
        """Get from json file containing local variables."""
        data = utils.load_json(path)
        return cls.from_localsdict(data)

    @classmethod
    def from_localsdict(cls, data):
        """Get from local vars dictionary."""
        first_sections = infer_sections(data)
        assert len(first_sections) == 1
        section = first_sections.pop()

        data = strip_section(data, section)
        sections = infer_sections(data)
        data = build_sections(data, sections)
        return cls(data, section=section)

    @classmethod
    def from_toml(cls, path):
        data = utils.load_toml(path)
        section = data.pop("section")
        # transform each section
        for subsec in data:
            data[subsec] = to_int_naming(data[subsec], "selected_markers", "tube")
        return cls(data, section=section)

    def to_json(self, path):
        utils.save_json(self.dict, path)

    def to_toml(self, path):
        tomldata = self._data.copy()
        tomldata["section"] = self.section
        # transform sections with int keys to strings to avoid TOML errors
        for subsec in tomldata:
            tomldata[subsec] = to_string_naming(tomldata[subsec], "selected_markers", "tube")
        utils.save_toml(tomldata, path)

    def __getitem__(self, index):
        """Get data with the given index. Either return another configuration
        instance or return the single key."""
        if index in self._data:
            return self._data[index]
        if index.startswith(self.section):
            index = index[len(self.section) + 1:]
        section, key = index.split("_", 1)
        return self._data[section][key]

    def __setitem__(self, index, value):
        """Get the given section with the newly provided value."""
        self._data[index] = value

    def __repr__(self):
        return f"<{self.section}: {len(self._data)} subsections>"