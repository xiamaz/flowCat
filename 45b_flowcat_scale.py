from sklearn.preprocessing import StandardScaler
import fcsparser
testfile = "output/4-flowsom-cmp/dataset/fcsdata/159e4754f47453ea9f0d177d88592eb62af598d8-2 CLL 9F 01 N09 001.LMD"
meta, data = fcsparser.parse(testfile, encoding="latin-1")
result = StandardScaler().fit_transform(data)
print(result[0, :])
