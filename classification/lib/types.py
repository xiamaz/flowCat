'''Types information for data structures used in the classification process.'''
import typing

# Information for upsampled tubes with multiple parts
FilesDict = typing.Dict[str, typing.List[str]]
GroupSelection = list
GroupName = str
SizesDict = typing.Dict[GroupName, int]
SizeOption = typing.Union[int, SizesDict]
