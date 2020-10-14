user_count.most_common(10)
Out[2]: 
[('A3V6Z4RCDGRC44', 872),
 ('A3W4D8XOGLWUN5', 791),
 ('AJKWF4W7QD4NS', 784),
 ('A2QHS1ZCIQOL7E', 517),
 ('A2TCG2HV1VJP6V', 458),
 ('A29BQ6B90Y1R5F', 420),
 ('AFV2584U13XP3', 332),
 ('A20DZX38KRBIT8', 315),
 ('A2582KMXLK2P06', 261),
 ('A74TA8X5YQ7NE', 260)]

item_count.most_common(10)
Out[3]: 
[('B00BGA9WK2', 6519),
 ('B002VBWIP6', 4441),
 ('B0015AARJI', 3916),
 ('B00178630A', 3079),
 ('B000FKBCX4', 2800),
 ('B000B9RI14', 2658),
 ('B007VTVRFA', 2647),
 ('B007FTE2VW', 2369),
 ('B003VANOFY', 2178),
 ('B0009VXBAQ', 2143)]


for d in tqdm(read_gzip(REVIEW_GZIP), "Second pass of reviews for \"{}\"".format(CATEGORY)):
    if "Yelp" in CATEGORY:
        user = d['user_id']
        item = d['business_id']
    else:
        user = d['reviewerID']
        item = d['asin']
    if hit((user, item), users_dict, items_dict):
        if "Yelp" in CATEGORY:
            rating = d['stars']
            date = d['date']
            date = int(time.mktime(time.strptime(date, "%Y-%m-%d")))
            
            vf = []
        else:
            rating = d['overall']
            date = d['unixReviewTime']
            text = simple_tokenizer(d['reviewText'])
            vf = item_features[item][0]
        interactions.append([user, item, rating, date, text, vf])


# ========== Construct the user & item visual features ===========
users_visuals = defaultdict(list)
items_visuals = defaultdict(list)

append_to_file(output_log, "\nConsolidating user/item visual features from TRAINING set")
for interaction in tqdm(train_interactions, desc="Consolidating user/item visual features from TRAINING set"):
    user = interaction[0]
    item = interaction[1]

    vf = interaction[5]

    users_visuals[user].append(vf)
    items_visuals[item].append(vf)

users_vis = defaultdict(list)
items_vis = defaultdict(list)

append_to_file(output_log, "\nCreating user visuals from TRAINING set")
for user, visuals in users_visuals.items():
    random.shuffle(visuals)
    for visual in visuals:
        currUserVisLen = len(users_vis[user])
        if currUserVisLen < MAX_VIS_LEN:
            users_vis[user].extend(visual[:(MAX_VIS_LEN - currUserDocLen)])
        else:
            break

append_to_file(output_log, "Creating item visuals from TRAINING set")
for item, visuals in items_visuals.items():
    random.shuffle(visuals)
    for visual in visuals:
        currItemVisLen = len(items_vis[item])
        if currItemVisLen < MAX_VIS_LEN:
            items_vis[item].extend(visual[:(MAX_VIS_LEN - currItemVisLen)])
        else:
            break

# Force garbage collection
del users_visuals
del items_visuals
gc.collect()

# Just checking the visual features length
minUserVisLen = MAX_VIS_LEN
minItemVisLen = MAX_VIS_LEN

for user, user_vis in users_vis.items():
    currUserVisLen = len(user_vis)
    minUserVisLen = min(minUserVisLen, currUserVisLen)

for item, item_vis in items_vis.items():
    currItemVisLen = len(item_vis)
    minItemVisLen = min(minItemVisLen, currItemVisLen)

append_to_file(output_log, "\nMinimum User Vis Len: {}, Minimum Item Vis Len: {}".format(minUserVisLen, minItemVisLen))
# ========== Construct the user & item visual features ===========