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

# Notes of ANR

### ANR.py

This is the complete Aspect-based Neural Recommender (ANR), with ARL and AIE as its main components.

```python
class ANR(nn.Module):
    def __init__(self, logger, args, num_users, num_items):
        super(ANR, self).__init__()
        self.logger = logger
        self.args = args
        self.num_users = num_users
        self.num_items = num_items

        # User Documents & Item Documents (Input)
        self.uid_userDoc = nn.Embedding(self.num_users, self.args.max_doc_len)              # num_users x max_doc_len
        self.uid_userDoc.weight.requires_grad = False

        self.iid_itemDoc = nn.Embedding(self.num_items, self.args.max_doc_len)              # num_items x max_doc_len
        self.iid_itemDoc.weight.requires_grad = False

        # Word Embeddings (Input)
        self.wid_wEmbed = nn.Embedding(self.args.vocab_size, self.args.word_embed_dim)      # vocab_size x word_embed_dim
        self.wid_wEmbed.weight.requires_grad = False

        # Aspect Representation Learning - Single Aspect-based Attention Network (Shared between User & Item)
        self.shared_ANR_ARL = ANR_ARL(logger, args)

        # Rating Prediction - Aspect Importance Estimation + Aspect-based Rating Prediction
        if self.args.model == "ANR":
            # Aspect-Based Co-Attention (Parallel Co-Attention, using the Affinity Matrix as a Feature)
            # Aspect Importance Estimation
            self.ANR_AIE = ANR_AIE(logger, args)

            # Aspect-Based Rating Predictor based on the estimated Aspect-Level Importance
            self.ANR_RatingPred = ANR_RatingPred(logger, args, self.num_users, self.num_items)

        # 'Simplified Model' - Basically, ARL + simplfied network (3x FCs) for rating prediction
        # The only purpose of this is to obtain the pretrained weights for ARL
        elif self.args.model == "ANRS":
            # Rating Predictor using the 'Simplified Model'
            self.ANRS_RatingPred = ANRS_RatingPred(logger, args)

    def forward(self, batch_uid, batch_iid, verbose=0):
        # Input
        batch_userDoc = self.uid_userDoc(batch_uid)										# bsz x max_doc_len
        batch_itemDoc = self.iid_itemDoc(batch_iid)										# bsz x max_doc_len

        # Embedding Layer
        batch_userDocEmbed = self.wid_wEmbed(batch_userDoc.long())						# bsz x max_doc_len x word_embed_dim
        batch_itemDocEmbed = self.wid_wEmbed(batch_itemDoc.long())						# bsz x max_doc_len x word_embed_dim

        # =========== User Aspect-Based Representations ===========
        # userAspAttn:	bsz x num_aspects x max_doc_len
        # userAspDoc:	bsz x num_aspects x h1
        userAspAttn, userAspDoc = self.shared_ANR_ARL(batch_userDocEmbed, verbose=verbose)
        # =========== User Aspect-Based Representations ===========

        # =========== Item Aspect-Based Representations ===========
        # itemAspAttn:	bsz x num_aspects x max_doc_len
        # itemAspDoc:	bsz x num_aspects x h1
        itemAspAttn, itemAspDoc = self.shared_ANR_ARL(batch_itemDocEmbed, verbose=verbose)
        # =========== Item Aspect-Based Representations ===========

        if self.args.model == "ANR":
            # Aspect-based Co-Attention --- Aspect Importance Estimation
            userCoAttn, itemCoAttn = self.ANR_AIE(userAspDoc, itemAspDoc, verbose=verbose)

            # Aspect-Based Rating Predictor based on the estimated Aspect-Level Importance
            rating_pred = self.ANR_RatingPred(userAspDoc, itemAspDoc, userCoAttn, itemCoAttn, batch_uid, batch_iid, verbose=verbose)

        # 'Simplified Model' - Basically, ARL + simplfied network (3x FCs) for rating prediction
        # The only purpose of this is to obtain the pretrained weights for ARL
        elif self.args.model == "ANRS":
            # Rating Prediction using 3x FCs
            # rating_pred:	bsz x 1
            rating_pred = self.ANRS_RatingPred(userAspDoc, itemAspDoc, verbose=verbose)

        return rating_pred
```

### ANR_ARL.py

Aspect-based Representation Learning (ARL)

```python
class ANR_ARL(nn.Module):
    def __init__(self, logger, args):
        super(ANR_ARL, self).__init__()
        self.logger = logger
        self.args = args

        # Aspect Embeddings
        self.aspEmbed = nn.Embedding(self.args.num_aspects, self.args.ctx_win_size * self.args.h1)  # num_aspects x (ctx_win_size x h1)
        self.aspEmbed.weight.requires_grad = True

        # Aspect-Specific Projection Matrices, W_a
        self.aspProj = nn.Parameter(torch.Tensor(self.args.num_aspects, self.args.word_embed_dim, self.args.h1),
                                    requires_grad=True)                                             # num_aspects x word_embed_dim x h1

        # Initialize all weights using random uniform distribution from [-0.01, 0.01]
        self.aspEmbed.weight.data.uniform_(-0.01, 0.01)
        self.aspProj.data.uniform_(-0.01, 0.01)

    '''
    [Input]     batch_docIn:    bsz x max_doc_len x word_embed_dim
    [Output]    batch_aspRep:   bsz x num_aspects x h1
    '''
    def forward(self, batch_docIn, verbose=0):
        # Loop over all aspects
        lst_batch_aspAttn = []
        lst_batch_aspRep = []
        for a in range(self.args.num_aspects):
            # Aspect-Specific Projection of Input Word Embeddings
            # (bsz x max_doc_len x word_embed_dim) * (word_embed_dim x h1) -> bsz x max_doc_len x h1
            batch_aspProjDoc = torch.matmul(batch_docIn, self.aspProj[a])                           # bsz x max_doc_len x h1

            # Aspect Embedding: (bsz x (ctx_win_size x h1) x 1) after tranposing!
            bsz = batch_docIn.size()[0]
            batch_aspEmbed = self.aspEmbed(to_var(torch.LongTensor(bsz, 1).fill_(a), use_cuda=self.args.use_cuda))  # bsz x 1 x (ctx_win_size x h1)
            batch_aspEmbed = torch.transpose(batch_aspEmbed, 1, 2)                                                  # bsz x (ctx_win_size x h1) x 1

            # Window Size (self.args.ctx_win_size) of 1: Calculate Attention based on the word itself!
            if self.args.ctx_win_size == 1:
                # Calculate Attention: Inner Product & Softmax
                # (bsz x max_doc_len x h1) x (bsz x h1 x 1) -> (bsz x max_doc_len x 1)
                batch_aspAttn = torch.matmul(batch_aspProjDoc, batch_aspEmbed)
                batch_aspAttn = F.softmax(batch_aspAttn, dim=1)
            # Context-based Word Importance
            # Calculate Attention based on the word itself, and the (self.args.ctx_win_size - 1) / 2 word(s) before & after it
            else:
                # Pad the document
                pad_size = int((self.args.ctx_win_size - 1) / 2)
                batch_aspProjDoc_padded = F.pad(batch_aspProjDoc, (0, 0, pad_size, pad_size), "constant", 0)		# bsz x (max_doc_len+2) x h1

                # Use "sliding window" using stride of 1 (word at a time) to generate word chunks of ctx_win_size
                # (bsz x max_doc_len x h1) -> (bsz x max_doc_len x (ctx_win_size x h1))
                batch_aspProjDoc_padded = batch_aspProjDoc_padded.unfold(1, self.args.ctx_win_size, 1)				# bsz x max_doc_len x h1 x ctx_win_size
                batch_aspProjDoc_padded = torch.transpose(batch_aspProjDoc_padded, 2, 3)							# bsz x max_doc_len x ctx_win_size x h1
                batch_aspProjDoc_padded = batch_aspProjDoc_padded.contiguous().view(-1, self.args.max_doc_len,		# bsz x max_doc_len x (ctx_win_size x h1)
                                                                                    self.args.ctx_win_size * self.args.h1)

                # Calculate Attention: Inner Product & Softmax
                # (bsz x max_doc_len x (ctx_win_size x h1)) * (bsz x (ctx_win_size x h1) x 1) -> (bsz x max_doc_len x 1)
                batch_aspAttn = torch.matmul(batch_aspProjDoc_padded, batch_aspEmbed)								# bsz x max_doc_len x 1

                batch_aspAttn = F.softmax(batch_aspAttn, dim=1)														# bsz x max_doc_len x 1

            # Weighted Sum: Broadcasted Element-wise Multiplication & Sum over Words
            # (bsz x max_doc_len x h1) * (bsz x max_doc_len x h1) -> (bsz x h1)
            batch_aspRep = batch_aspProjDoc * batch_aspAttn.expand_as(batch_aspProjDoc)								# bsz x max_doc_len x h1   
            # bsz x max_doc_len x h1 -> bsz x h1
            batch_aspRep = torch.sum(batch_aspRep, dim=1)															# bsz x h1   

            # Store the results (Attention & Representation) for this aspect
            lst_batch_aspAttn.append(torch.transpose(batch_aspAttn, 1, 2))              							# bsz x 1 x max_doc_len
            lst_batch_aspRep.append(torch.unsqueeze(batch_aspRep, 1))                   							# bsz x 1 x h1

        # Reshape the Attentions & Representations
        # batch_aspAttn:        (bsz x num_aspects x max_doc_len)
        # batch_aspRep:         (bsz x num_aspects x h1)
        batch_aspAttn = torch.cat(lst_batch_aspAttn, dim=1)
        batch_aspRep = torch.cat(lst_batch_aspRep, dim=1)

        # Returns the aspect-level attention over document words, and the aspect-based representations
        return batch_aspAttn, batch_aspRep

```

### ANR_AIE.py

Aspect Importance Estimation (AIE)

```python
class ANR_AIE(nn.Module):
    def __init__(self, logger, args):
        super(ANR_AIE, self).__init__()
        self.logger = logger
        self.args = args

        # Matrix for Interaction between User Aspect-level Representations & Item Aspect-level Representations 
        # This is a learnable (h1 x h1) matrix, i.e. User Aspects - Rows, Item Aspects - Columns
        self.W_a = nn.Parameter(torch.Tensor(self.args.h1, self.args.h1), requires_grad=True)   # W_s: h1 x h1

        self.W_u = nn.Parameter(torch.Tensor(self.args.h2, self.args.h1), requires_grad=True)   # W_x: h2 x h1
        self.w_hu = nn.Parameter(torch.Tensor(self.args.h2, 1), requires_grad=True)             # v_x: h2 x 1

        self.W_i = nn.Parameter(torch.Tensor(self.args.h2, self.args.h1), requires_grad=True)   # W_y: h2 x h1
        self.w_hi = nn.Parameter(torch.Tensor(self.args.h2, 1), requires_grad=True)             # v_y: h2 x 1

        # Initialize all weights using random uniform distribution from [-0.01, 0.01]
        self.W_a.data.uniform_(-0.01, 0.01)
        self.W_u.data.uniform_(-0.01, 0.01)
        self.w_hu.data.uniform_(-0.01, 0.01)
        self.W_i.data.uniform_(-0.01, 0.01)
        self.w_hi.data.uniform_(-0.01, 0.01)

    '''
    [Input]  userAspRep: bsz x num_aspects x h1 P_u
    [Input]  itemAspRep: bsz x num_aspects x h1 Q_i
    '''
    def forward(self, userAspRep, itemAspRep, verbose=0):
        userAspRepTrans = torch.transpose(userAspRep, 1, 2)							# bsz x h1 x num_aspects
        itemAspRepTrans = torch.transpose(itemAspRep, 1, 2)							# bsz x h1 x num_aspects       

        '''
        Affinity Matrix (User Aspects x Item Aspects), i.e. User Aspects - Rows, Item Aspects - Columns
        S = RELU(P_u * W_s * Q_i^T)
        '''
        # (bsz x num_aspects x h1) * (h1 x h1) -> bsz x num_aspects x h1
        affinityMatrix = torch.matmul(userAspRep, self.W_a)							# bsz x num_aspects x h1      

        # (bsz x num_aspects x h1) * (bsz x h1 x num_aspects) -> bsz x num_aspects x num_aspects
        affinityMatrix = torch.matmul(affinityMatrix, itemAspRepTrans)				# bsz x num_aspects x num_aspects       

        # Non-Linearity: ReLU
        affinityMatrix = F.relu(affinityMatrix)										# bsz x num_aspects x num_aspects

        '''
        H_u = RELU(P_u * W_x + S^T * (Q_i * W_y))
        beta_u = softmax(H_u * v_x)
        '''
        # =========== User Importance (over Aspects) ===========
        # (h2 x h1) * (bsz x h1 x num_aspects) -> bsz x h2 x num_aspects
        H_u_1 = torch.matmul(self.W_u, userAspRepTrans)								# bsz x h2 x num_aspects
        H_u_2 = torch.matmul(self.W_i, itemAspRepTrans)								# bsz x h2 x num_aspects

        # (bsz x h2 x num_aspects) * (bsz x num_aspects x num_aspects) -> bsz x h2 x num_aspects
        H_u_2 = torch.matmul(H_u_2, affinityMatrix)									# bsz x h2 x num_aspects
        H_u = H_u_1 + H_u_2                                                         # bsz x h2 x num_aspects

        # Non-Linearity: ReLU
        H_u = F.relu(H_u)                                                           # bsz x h2 x num_aspects

        # User Aspect-level Importance
        # (1 x h2) * (bsz x h2 x num_aspects) -> bsz x 1 x num_aspects
        userAspImpt = torch.matmul(torch.transpose(self.w_hu, 0, 1), H_u)           # bsz x 1 x num_aspects

        userAspImpt = torch.transpose(userAspImpt, 1, 2)							# bsz x num_aspects x 1

        userAspImpt = F.softmax(userAspImpt, dim=1)                                 # bsz x num_aspects x 1

        userAspImpt = torch.squeeze(userAspImpt, 2)									# bsz x num_aspects, beta_u
        # =========== User Importance (over Aspects) ===========

        '''
        H_i = RELU(Q_i * W_y + S * (P_u * W_x))
        beta_i = softmax(H_i * v_y)
        '''
        # =========== Item Importance (over Aspects) ===========
        # (h2 x h1) * (bsz x h1 x num_aspects) -> bsz x h2 x num_aspects
        H_i_1 = torch.matmul(self.W_i, itemAspRepTrans)								# bsz x h2 x num_aspects
        H_i_2 = torch.matmul(self.W_u, userAspRepTrans)								# bsz x h2 x num_aspects

        # (bsz x h2 x num_aspects) * (bsz x num_aspects x num_aspects) -> bsz x h2 x num_aspects
        H_i_2 = torch.matmul(H_i_2, torch.transpose(affinityMatrix, 1, 2))			# bsz x h2 x num_aspects
        H_i = H_i_1 + H_i_2                                                         # bsz x h2 x num_aspects

        # Non-Linearity: ReLU
        H_i = F.relu(H_i)                                                           # bsz x h2 x num_aspects

        # Item Aspect-level Importance
        # (1 x h2) * (bsz x h2 x num_aspects) -> bsz x 1 x num_aspects
        itemAspImpt = torch.matmul(torch.transpose(self.w_hi, 0, 1), H_i)           # bsz x 1 x num_aspects

        itemAspImpt = torch.transpose(itemAspImpt, 1, 2)							# bsz x num_aspects x 1

        itemAspImpt = F.softmax(itemAspImpt, dim=1)                                 # bsz x num_aspects x 1

        itemAspImpt = torch.squeeze(itemAspImpt, 2)									# bsz x num_aspects, beta_i
        # =========== Item Importance (over Aspects) ===========
        return userAspImpt, itemAspImpt
```

