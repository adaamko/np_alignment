import os
import sys
import xml.etree.ElementTree as et

# extracts NPs from SMULTRON treebank files

def main(treebank_dir):
    main_dict = {}
    for dirname in os.listdir(treebank_dir):
        dirname = os.path.join(treebank_dir, dirname)
        for filename in os.listdir(dirname):
            if filename.startswith("smultron"):
                if "xml" not in filename:
                    continue
                file_id = filename.split("_")[-1]
                lang = filename.split("_")[1]
                # print(lang)
            
                if file_id not in main_dict:
                    main_dict[file_id] = {}
                
                tree = et.parse(os.path.join(dirname, filename))
                root = tree.getroot()
                
                for s in root.iter("s"):
                    sentence_id = s.get("id")
                    words_dict = {}
                    for t in s.iter("t"):
                        words_dict[t.get("id")] = t.get("word")
                                                              
                    if sentence_id not in main_dict[file_id]:
                        main_dict[file_id][sentence_id] = {}
                    if lang not in main_dict[file_id][sentence_id]:
                        main_dict[file_id][sentence_id][lang] = []

                    np_id_list = []
                    phrase_dict = {}
                    for nt in s.iter("nt"):
                        phrase_id = nt.get("id")
                        if nt.get("cat") == "NP":
                            np_id_list.append(phrase_id)

                        phrase_list = []
                        for edge in nt.iter("edge"):
                            phrase_list.append(edge.get("idref"))
                        
                        phrase = {
                            "text": None,
                            "ids": phrase_list,
                        }
                        phrase_dict[phrase_id] = phrase

                    while True:
                        ready = True
                        for phrase_id, phrase in phrase_dict.items():
                            parts = []
                            phrase_ready = True
                            for part_id in phrase["ids"]:
                                if part_id not in words_dict:
                                    phrase_ready = False
                                    break
                                else:
                                    parts.append(words_dict[part_id])
                            
                            if phrase_ready:
                                words_dict[phrase_id] = " ".join(parts)
                            else:
                                ready = False
                        
                        # Break out of the infinite loop if everything was found
                        if ready:
                            break
                    
                    # Finally, store the NPs
                    for np_id in np_id_list:
                        main_dict[file_id][sentence_id][lang].append(words_dict[np_id])
                            

    print(main_dict)

main(sys.argv[1])
