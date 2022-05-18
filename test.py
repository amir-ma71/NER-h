
from langdetect import detect

final = {'Person': [":"], 'Organization': ['אגף ם', "fghndf"], 'Geo-Political': [], 'Location': [], 'Facility': [], 'Event': [], 'Product': [], 'Language': []}
deleted_keys = []
for ner in final.keys():
    if len(final[ner]) == 0:
        deleted_keys.append(ner)
    else:
        for w in final[ner]:
            try:
                if detect(w) != "he":
                    final[ner].remove(w)
            except:
                final[ner].remove(w)

for j in deleted_keys:
    del final[j]

print(final)