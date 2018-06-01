'''Types information for data structures used in the classification process.'''
import typing

# Information for upsampled tubes with multiple parts
FilesDict = typing.Dict[int, typing.Dict[int, str]]
GroupSelection = list
GroupName = str
SizesDict = typing.Dict[GroupName, int]
SizeOption = typing.Union[int, SizesDict]
MaybeList = typing.Union[typing.List[int], None]
