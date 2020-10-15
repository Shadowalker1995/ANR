# Notes of preprocessing_simple.py

### Saving Files

- `CATEGORY_env.pkl`: num_users, num_items, num_ratings, train_size, dev_size, test_size
- `CATEGORY_info.pkl`: words_count, word_wid, user_uid, item_iid, uid_userDocLen, iid_itemDocLen, uid_userVisLen, iid_itemVisLen
- `CATEGORY_interactions.pkl`
- `CATEGORY_train_interactions.pkl`
- `CATEGORY_dev_interactions.pkl`
- `CATEGORY_test_interactions.pkl`
- `CATEGORY_split_train.pkl`: a list, [(uid, iid, rating), ...]
- `CATEGORY_split_dev.pkl`
- `CATEGORY_split_test.pkl`
- `CATEGORY_uid_userDoc.npy`: a numpy array, row index: `uid`, row: `userDoc` in np.float dtype
- `CATEGORY_iid_itemDoc.npy`
- `CATEGORY_uid_userVis.npy`
- `CATEGORY_iid_itemVis.npy`

### Main processes

1. Initial pass of reviews to get the user-item interactions

    ```python
    # INPUT: REVIEW_GZIP = review_CATEGORY.json.gz
    interactions = []
    append_to_file(output_log, "\nInitial pass of reviews to get the user-item interactions!")
    for d in tqdm(read_gzip(REVIEW_GZIP), "Initial pass of reviews for \"{}\"".format(CATEGORY)):
        user = d['reviewerID']
        item = d['asin']
        interactions.append([user, item])
    ```

    

2. Second pass of visual features to get the item-feature interactions

    ```python
    # INPUT: VISUAL_JSON = image_feature_CATEGORY_pca.json
    item_features = defaultdict(list)
    with codecs.open(VISUAL_JSON, 'r', encoding='utf-8', errors='ignore') as inFile:
        lines = inFile.readlines()
        for line in tqdm(lines, "Second pass of visual features for \"{}\"".format(CATEGORY)):
            d = json.loads(line)
            item = d['asin']
            feature = d['feature']
            item_features[item].append(feature)
    item_feature_count = feature_count(item_features)
    lines.clear()
    gc.collect()
    ```

    

3. filter away users & items based on the num of images(`MIN_IMAGES` = 1)

    ```python
    interactions = [interaction for interaction in tqdm(interactions, "Filtering interactions") if interaction[1] in item_feature_count]
    ```

    

4. filter away users & items based on the num of reviews(`MIN_REVIEWS` = 1)

    ```python
    # Drop users with less than "MIN_REVIEWS" reviews
    drop_if_lt(user_count, MIN_REVIEWS)
    # Drop items with less than "MIN_REVIEWS" reviews
    drop_if_lt(item_count, MIN_REVIEWS)
    del interactions
    gc.collect()
    ```

    

5. Third pass of reviews to get the rating, date, the num of tokenized review and index

    ```python
    interactions = []
    index = 0
    for d in tqdm(read_gzip(REVIEW_GZIP), "Third pass of len of reviews for \"{}\"".format(CATEGORY)):
        user = d['reviewerID']
        item = d['asin']
        if hit((user, item), users_dict, items_dict):
            rating = d['overall']
            date = d['unixReviewTime']
            text_len = len(simple_tokenizer(d['reviewText']))
            interactions.append([user, item, rating, date, text_len, index])
        index += 1
    ```

    

6. filter user-item interactions based on minimum review length(`MIN_REVIEW_LEN` = 10)

    ```python
    interactions = [interaction for interaction in tqdm(interactions, "Filtering interactions") if (interaction[4] >= MIN_REVIEW_LEN)]
    # Starting to filter away users & items based on thresold of {MIN_REVIEWS} reviews after removing reviews with <= {MIN_REVIEW_LEN} tokens
    ```

    

7. For LIMITING large datasets

    ```python
    # Selecting a random subsample of {args.dataset_maximum_size} user-item interactions!
    len_interactions = len(interactions)
    if len_interactions > args.dataset_maximum_size:
        interactions = random.sample(interactions, args.dataset_maximum_size)
        
    # Sort interactions with the user-item pair index
    interactions = sorted(interactions, key=lambda x: x[5], reverse=False)
    
    # Get the real review text based on the index
    index = 0
    index_interation = 0
    for d in tqdm(read_gzip(REVIEW_GZIP), "Fourth pass of reviews for \"{}\"".format(CATEGORY)):
        try:
            if index == interactions[index_interation][5]:
                item = d['asin']
                text = simple_tokenizer(d['reviewText'])
                vf = item_features[item][0]
                interactions[index_interation] = interactions[index_interation][:4]
                interactions[index_interation].append(text)
                interactions[index_interation].append(vf)
                index_interation += 1
    	except IndexError as e:
            break
        index += 1
    ```

    

8. Saving `interactions` files(**optional**)

    ```python
    # Interactions
    # output_interactions: CATEGORY_interactions.pkl
    with open(output_interactions, "wb+") as f:
    	pickle.dump(interactions, f, protocol=2)
    ```

    

9. Split TRAIN/DEV/TEST dataset

    ```python
    # Random shuffle for all user-item interactions
    random.shuffle(interactions)
    # Total number of reviews
    num_reviews = len(interactions)
    
    train_index = int(args.train_ratio * num_reviews)
    train_interactions = interactions[:train_index]
    dev_ratio = args.train_ratio + dev_test_ratio
    dev_index = int(dev_ratio * num_reviews)
    dev_interactions = interactions[train_index:dev_index]
    test_interactions = interactions[dev_index:]
    
    del interactions
    gc.collect()
    
    # Assertion to make sure TRAIN + DEV + TEST=ORIGINAL DATASET SIZE
    assert len(train_interactions) + len(dev_interactions) + len(test_interactions) == num_reviews, "Train/Dev/Test split is wrong!!"
    ```

    

10. Remove users & items who do not appear in training set, from the dev and test sets

    ```python
    if args.dev_test_in_train:
        # Remove users & items who do not appear in training set, from the dev and test sets
        train_user_count, train_item_count = count(train_interactions)
        train_users_dict = dict(train_user_count)
        train_items_dict = dict(train_item_count)
        dev_interactions = [i for i in tqdm(dev_interactions, "Updating DEV interactions") if hit(i, train_users_dict, train_items_dict)]
        test_interactions = [i for i in tqdm(test_interactions, "Updating TEST interactions") if hit(i, train_users_dict, train_items_dict)]
    ```

    

11. Saving `train_interactions`, `dev_interactions`, `test_interactions` files

    ```python
    # output_train_interactions: CATEGORY_train_interactions.pkl
    with open(output_train_interactions, "wb+") as f:
        pickle.dump(train_interactions, f, protocol=2)
        
    # output_dev_interactions: CATEGORY_dev_interactions.pkl
    with open(output_dev_interactions, "wb+") as f:
        pickle.dump(dev_interactions, f, protocol=2)
        
    # output_test_interactions: CATEGORY_test_interactions.pkl
    with open(output_test_interactions, "wb+") as f:
        pickle.dump(test_interactions, f, protocol=2)
    ```

    

12. Construct the user & item documents

    ```python
    users_reviews = defaultdict(list)
    items_reviews = defaultdict(list)
    for interaction in tqdm(train_interactions, desc="Consolidating user/item reviews from TRAINING set"):
        user = interaction[0]
        item = interaction[1]
        text = interaction[4]
        users_reviews[user].append(text)
        items_reviews[item].append(text)
    
    # MAX_DOC_LEN = 500
    users_doc = defaultdict(list)
    items_doc = defaultdict(list)
    for user, reviews in users_reviews.items():
        random.shuffle(reviews)
        for review in reviews:
            currUserDocLen = len(users_doc[user])
            if currUserDocLen < MAX_DOC_LEN:
                users_doc[user].extend(review[:(MAX_DOC_LEN - currUserDocLen)])
            else:
                break
    
    for item, reviews in items_reviews.items():
        random.shuffle(reviews)
        for review in reviews:
            currItemDocLen = len(items_doc[item])
            if currItemDocLen < MAX_DOC_LEN:
                items_doc[item].extend(review[:(MAX_DOC_LEN - currItemDocLen)])
            else:
                break
                
    del users_reviews
    del items_reviews
    gc.collect()
    ```

    

13. Convert user & item documents to corresponding wid

    ```python
    # Get vocabulary based on the user & item documents
    words = []
    for user, user_doc in users_doc.items():
        words += [word for word in user_doc]
    for item, item_doc in items_doc.items():
        words += [word for word in item_doc]
        
    # Get unique words to form the vocabulary
    # NOTE: Word order in "words" is relative to frequency, i.e. Most frequent, 2nd-most frequent, and so on..
    # VOCAB_SIZE = 50000
    words_count = Counter(words).most_common(VOCAB_SIZE)
    words = [w for w, c in words_count]
    
    # Build word dictionary - PAD is 0, UNK (OOV) is 1
    word_wid = {word: (wid + 2) for wid, word in enumerate(words)}
    word_wid[PAD] = 0
    word_wid[UNK] = 1
    del words
    gc.collect()
    
    # Build user/item dictionary
    user_uid = {user: uid for uid, user in enumerate(users_doc.keys())}
    item_iid = {item: iid for iid, item in enumerate(items_doc.keys())}
    
    # Convert each user/item to its correspoding index in the user/item dictionary
    # Convert each word to its corresponding index in the word dictionary
    # print("For each user/item doc, converting words to wids using word_wid...\n")
    uid_userDoc = {user_uid[user]: doc2id(userDoc, word_wid) for user, userDoc in tqdm(users_doc.items(), desc="For each user doc, converting words to wids using word_wid...")}
    iid_itemDoc = {item_iid[item]: doc2id(itemDoc, word_wid) for item, itemDoc in tqdm(items_doc.items(), desc="For each item doc, converting words to wids using word_wid...")}
    del users_doc
    del items_doc
    gc.collect()
    
    # Store the actual length of each user/item document (before padding)
    uid_userDocLen = {uid: len(userDoc) for uid, userDoc in tqdm(uid_userDoc.items(), desc="Store the actual length of each user document (before padding)")}
    iid_itemDocLen = {iid: len(itemDoc) for iid, itemDoc in tqdm(iid_itemDoc.items(), desc="Store the actual length of each item document (before padding)")}
    
    # Pad the user/item documents to MAX_DOC_LEN
    uid_userDoc = {uid: post_padding(userDoc, MAX_DOC_LEN, word_wid[PAD]) for uid, userDoc in tqdm(uid_userDoc.items(), desc="Pad the user documents to MAX_DOC_LEN")}
    iid_itemDoc = {iid: post_padding(itemDoc, MAX_DOC_LEN, word_wid[PAD]) for iid, itemDoc in tqdm(iid_itemDoc.items(), desc="Pad the item documents to MAX_DOC_LEN")}
    ```

    

14. Prepare the `train/dev/test` sets and saving the splited sets and `info` files

    ```python
    # Convert user string to id, and item string to id
    # then constrct a list: [(uid, iid, rating), ...]
    train = prepare_set(train_interactions, user_uid, item_iid, uid_userDocLen, iid_itemDocLen, "TRAINING", output_log)
    dev = prepare_set(dev_interactions, user_uid, item_iid, uid_userDocLen, iid_itemDocLen, "DEV", output_log)
    test = prepare_set(test_interactions, user_uid, item_iid, uid_userDocLen, iid_itemDocLen, "TESTING", output_log)
    
    # Some useful information
    info = {
        'num_users': len(user_uid),
        'num_items': len(item_iid),
        'num_ratings': (len(train_interactions) + len(dev_interactions) + len(test_interactions)),
        'train_size': len(train_interactions),
        'dev_size': len(dev_interactions),
        'test_size': len(test_interactions)
    }
    # output_info: CATEGORY_info.pkl
    with open(output_info, "wb+") as f:
        pickle.dump(info, f, protocol=2)
    
    del train_interactions
    del dev_interactions
    del test_interactions
    del info
    gc.collect()
    
    # output_split_train: CATEGORY_split_train.pkl
    with open(output_split_train, "wb+") as f:
        pickle.dump(train, f, protocol=2)
    # output_split_dev: CATEGORY_split_dev.pkl
    with open(output_split_dev, "wb+") as f:
        pickle.dump(dev, f, protocol=2)
    # output_split_dev: CATEGORY_split_test.pkl
    with open(output_split_test, "wb+") as f:
        pickle.dump(test, f, protocol=2)
    del train
    del dev
    del test
    gc.collect()
    ```

    ```python
    def prepare_set(interactions, user_uid, item_iid, uid_userDocLen, iid_itemDocLen, set_type, output_log, printToScreen=False):
        lst_uid = []
        lst_iid = []
        lst_rating = []
    
        for interaction in tqdm(interactions, "Preparing the {} set".format(set_type)):
            user = interaction[0]
            item = interaction[1]
            rating = interaction[2]
    
            # Convert user string to id, and item string to id
            uid = user_uid[user]
            iid = item_iid[item]
    
            lst_uid.append(uid)
            lst_iid.append(iid)
            lst_rating.append(rating)
        return zip(lst_uid, lst_iid, lst_rating)
    ```

    

15. Construct the user & item documents matrix and saving `uid_userDoc` & `iid_itemDoc` files

    ```python
    # output_uid_userDoc: CATEGORY_uid_userDoc.npy
    uid_userDoc_Matrix = createNumpyMatrix(0, len(uid_userDoc), uid_userDoc)
    np.save(output_uid_userDoc, uid_userDoc_Matrix)
    
    # output_iid_itemDoc: CATEGORY_iid_itemDoc.npy
    iid_itemDoc_Matrix = createNumpyMatrix(0, len(iid_itemDoc), iid_itemDoc)
    np.save(output_iid_itemDoc, iid_itemDoc_Matrix)
    
    del uid_userDoc
    del iid_itemDoc
    del uid_userDoc_Matrix
    del iid_itemDoc_Matrix
    gc.collect()
    ```

    ```python
    def createNumpyMatrix(startIndex, endIndex, mapping):
        npMatrix = []
        rows = (endIndex - startIndex)
        columns = len(mapping[startIndex])
        for idx in range(startIndex, endIndex):
            vec = mapping[idx]
            npMatrix.append(vec)
        npMatrix = np.stack(npMatrix)
        npMatrix = np.reshape(npMatrix, (rows, columns))
        return npMatrix
    ```

    

16. Construct the user & item visual features

    ```python
    # load train_interactions
    train_interactions = load_pickle(output_train_interactions)
    
    users_visuals = defaultdict(list)
    items_visuals = defaultdict(list)
    for interaction in tqdm(train_interactions, desc="Consolidating user/item visual features from TRAINING set"):
        user = interaction[0]
        item = interaction[1]
        vf = interaction[5]
        users_visuals[user].append(vf)
        items_visuals[item].append(vf)
    
    del train_interactions
    gc.collect()
    
    users_vis = defaultdict(list)
    items_vis = defaultdict(list)
    # MAX_VIS_LEN = 500
    for user, visuals in users_visuals.items():
        random.shuffle(visuals)
        for visual in visuals:
            currUserVisLen = len(users_vis[user])
            if currUserVisLen < MAX_VIS_LEN:
                users_vis[user].extend(visual[:(MAX_VIS_LEN - currUserVisLen)])
            else:
                break
    
    for item, visuals in items_visuals.items():
        random.shuffle(visuals)
        for visual in visuals:
            currItemVisLen = len(items_vis[item])
            if currItemVisLen < MAX_VIS_LEN:
                items_vis[item].extend(visual[:(MAX_VIS_LEN - currItemVisLen)])
            else:
                break
    
    del users_visuals
    del items_visuals
    gc.collect()
    ```

    

17. Convert user & item to uid & iid

    ```python
    # print("Convert user & item to uid & iid...\n")
    uid_userVis = {user_uid[user]: userVis for user, userVis in tqdm(users_vis.items(), desc="Convert user to uid...")}
    iid_itemVis = {item_iid[item]: itemVis for item, itemVis in tqdm(items_vis.items(), desc="Convert item to iid...")}
    del users_vis
    del items_vis
    gc.collect()
    
    # Store the actual length of each user/item visual feature (before padding)
    uid_userVisLen = {uid: len(userVis) for uid, userVis in tqdm(uid_userVis.items(), desc="Store the actual length of each user visual feature (before padding)")}
    iid_itemVisLen = {iid: len(itemVis) for iid, itemVis in tqdm(iid_itemVis.items(), desc="Store the actual length of each item visual feature (before padding)")}
    
    # Pad the user/item visual feature to MAX_VIS_LEN
    uid_userVis = {uid: post_padding(userVis, MAX_VIS_LEN, 0) for uid, userVis in tqdm(uid_userVis.items(), desc="Pad the user visual feature to MAX_VIS_LEN")}
    iid_itemVis = {iid: post_padding(itemVis, MAX_VIS_LEN, 0) for iid, itemVis in tqdm(iid_itemVis.items(), desc="Pad the item visual feature to MAX_VIS_LEN")}
    ```

    

18. Construct the user & item visual features matrix and saving `uid_userVis` & `iid_itemVis` files

    ```python
    # output_uid_userVis: CATEGORY_uid_userVis.npy
    append_to_file(output_log, "\nCreating numpy matrix for uid_userVis..")
    uid_userVis_Matrix = createNumpyMatrix(0, len(uid_userVis), uid_userVis)
    np.save(output_uid_userVis, uid_userVis_Matrix)
    
    # output_iid_itemVis: CATEGORY_iid_itemVis.npy
    append_to_file(output_log, "\nCreating numpy matrix for iid_itemVis..")
    iid_itemVis_Matrix = createNumpyMatrix(0, len(iid_itemVis), iid_itemVis)
    np.save(output_iid_itemVis, iid_itemVis_Matrix)
    
    del uid_userVis_Matrix
    del iid_itemVis_Matrix
    del uid_userVis
    del iid_itemVis
    gc.collect()
    ```

    

19. Saving `env` files

    ```python
    env = {
        # List of (word, frequency)
        'words_count': words_count,
        # Mapping from word to wid in the word dictionary
        'word_wid': word_wid,
        # Mapping from user to uid in the user dictionary (More for finding user based on uid)
        'user_uid': user_uid,
        # Mapping from item to iid in the item dictionary (More for finding item based on iid)
        'item_iid': item_iid,
        # Mapping from uid to the original userDocLen
        'uid_userDocLen': uid_userDocLen,
        # Mapping from iid to the original itemDocLen
        'iid_itemDocLen': iid_itemDocLen,
        # Mapping from uid to the original userVisLen
        'uid_userVisLen': uid_userVisLen,
        # Mapping from iid to the original itemVisLen
        'iid_itemVisLen': iid_itemVisLen
    }
    
    # output_env: CATEGORY_env.pkl
    with open(output_env, "wb+") as f:
        pickle.dump(env, f, protocol=2)
    
    del env
    gc.collect()
    ```

# Notes of preprocessing_vector.py

### Saving Files

- `CATEGORY_wid_wordEmbed.npy`: a numpy array, row index: `wid`, row: `wordEmbed`

### Main processes

1. Load word-to-wid mappings from the environment file

    ```python
    # Load word-to-wid mappings from the environment file
    env = load_pickle( input_env )
    word_wid = env['word_wid']
    wid_word = {wid: word for word, wid in word_wid.items()}
    
    del env
    gc.collect()
    
    # Vocab
    vocab = word_wid.keys()
    ```

    

2. Word embedding

    ```python
    embeddings = {}
    # Load word embeddings
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(input_embeddings, binary=True)
    w2v_vocab = w2v_model.vocab.keys()
    w2v_vocab_dict = {word: "" for word in w2v_vocab}
    for v in tqdm(vocab, "Processing pretrained vectors"):
        try:
            _ = w2v_vocab_dict[v]
            embeddings[v] = w2v_model[v]
        except:
            pass
    
    wordEmbedMatrix = []
    noPretrainedEmb_words = []
    
    # For '<pad>' and '<unk>'
    # args.emb_dim = 300
    wordEmbedMatrix.append(np.zeros(args.emb_dim).tolist())
    wordEmbedMatrix.append(np.zeros(args.emb_dim).tolist())
    noPretrainedEmb_words.append(PAD)
    noPretrainedEmb_words.append(UNK)
    for wid in tqdm(range(2, len(word_wid)), "Forming Matrix"):
        word = wid_word[wid]
        try:
            vec = embeddings[word]
            vec = vec.tolist()
            vec = [float(x) for x in vec]
            wordEmbedMatrix.append(vec)
        except:
            # Words that do not have a pretrained embedding are initialized randomly using a uniform distribution U(−0.01, 0.01)
            vec = np.random.uniform(low=-args.emb_rand_init, high=args.emb_rand_init, size=args.emb_dim).tolist()
            wordEmbedMatrix.append(vec)
            noPretrainedEmb_words.append(word)
    
    wordEmbedMatrix = np.stack(wordEmbedMatrix)
    wordEmbedMatrix = np.reshape(wordEmbedMatrix, (len(word_wid), args.emb_dim))
    ```

    

3. Saving `wordEmbedMatrix` file

    ```python
    # output_wid_wordEmbed: CATEGORY_wid_wordEmbed.npy
    np.save(output_wid_wordEmbed, wordEmbedMatrix)
    ```

