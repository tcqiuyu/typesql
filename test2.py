import json
import torch
import datetime
import argparse
import nltk
import numpy as np
from typesql.utils import *
from typesql.model.sqlnet import SQLNet
from nltk.tokenize import TreebankWordTokenizer
from data_process_test import *


class Keys(object):
    # keys for question
    KG_ENTITIES = "kg_entities"
    QUESTION = "question"
    QUESTION_TOK_KGCOL = "question_tok_kgcol"
    QUESTION_TYPE_KGCOL = "question_type_kgcol"
    QUESTION_TYPE_KGCOL_LIST = "question_type_kgcol_list"
    QUESTION_TYPE_ORG_KGCOL = "question_type_org_kgcol"
    QUESTION_TOK_ORG = "question_tok_org"
    QUESTION_TABLE_ID = "table_id"
    QUESTION_TOK_CONCOL = "question_tok_concol"
    QUESTION_TYPE_CONCOL = "question_type_concol"
    QUESTION_TYPE_CONCOL_LIST = "question_type_concol_list"
    QUESTION_TYPE_ORG_CONCOL = "question_type_org_concol"
    QUESTION_TOK_SPACE = "question_tok_space"

    # keys for table
    TABLE_ID = "id"
    TABLE_ROWS = "rows"
    HEADER_TOK = "header_tok"
    HEADER_TYPE = "header_type_kg"

    # old keys
    QUESTION_TOK_TYPE = "question_tok_type"  # => q_type_kgcol_list
    QUESTION_TOK = "question_tok"  # => q_tok_kgcol

    # keys for meta information
    # meta header: {TYPE: TYPE_HEADER, META_CLS: ..., META_SIM_CLS: ...}
    # meta type: {TYPE: TYPE_TYPE, META_CLS: ..., META_SIM_CLS: ...}
    # meta kg: {TYPE: TYPE_KG, META_CLS: ..., META_SIM_CLS: ...}
    META = "meta"
    TYPE = "meta_type"
    TYPE_HEADER = "header"
    TYPE_TYPE = "type"
    TYPE_KG = "kg"
    TYPE_NONE = "none"
    META_CLS = "cls"
    META_SIM_CLS = "sim_cls"
    META_TOKS = "meta_tok"

    # keys for meta class
    META_DATE = "date"
    META_YEAR = "year"
    META_GAME_SCORE = "game score"
    META_INT = "integer"
    META_INT_SMALL = "small integer"
    META_INT_NEG = "negative integer"
    META_INT_MED = "medium integer"
    META_INT_BIG = "big integer"
    META_INT_LARGE = "large integer"
    META_FLOAT = "float"
    META_FLOAT_SMALL = "small float"
    META_FLOAT_NEG = "negative float"
    META_FLOAT_MED = "medium float"
    META_FLOAT_LARGE = "large float"

    META_LIST = ["person", "country", "place", "organization", "sport"]
    META_MAP = {
        META_DATE: META_DATE,
        META_YEAR: META_YEAR,
        META_GAME_SCORE: META_INT,
        META_INT_SMALL: META_INT,
        META_INT_NEG: META_INT,
        META_INT_MED: META_INT,
        META_INT_BIG: META_INT,
        META_INT_LARGE: META_INT,
        META_INT: META_INT,
        META_FLOAT: META_FLOAT,
        META_FLOAT_SMALL: META_FLOAT,
        META_FLOAT_NEG: META_FLOAT,
        META_FLOAT_MED: META_FLOAT,
        META_FLOAT_LARGE: META_FLOAT
    }
    META_MAP["person"] = "person"
    META_MAP["country"] = "country"
    META_MAP["place"] = "place"
    META_MAP["organization"] = "organization"
    META_MAP["sport"] = "sportsteam"

    # randomly chosen NONE string
    NONE = "te8r2ed"
    COLUMN = "column"
    ENTITY = "entity"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd", type=str, default='saved_model_con', help='set model save directory')
    parser.add_argument('--train_emb', action='store_true',
                        help='Use trained word embedding for SQLNet.')
    args = parser.parse_args()

    _, _, val_sql_data, val_table_data, \
    test_sql_data, test_table_data, \
    TRAIN_DB, DEV_DB, TEST_DB = load_dataset()

    word_emb = load_concat_wemb("glove/glove.42B.300d.txt",
                                "para-nmt-50m/paragram_sl999_czeng.txt")

    model = SQLNet(word_emb, N_word=600, gpu=False, trainable_emb=False, db_content=1)
    agg_m, sel_m, cond_m, agg_e, sel_e, cond_e = best_model_name(args)

    print "Loading from %s" % agg_m
    model.agg_pred.load_state_dict(torch.load(agg_m))
    print "Loading from %s" % sel_m
    model.selcond_pred.load_state_dict(torch.load(sel_m))
    print "Loading from %s" % cond_m
    model.op_str_pred.load_state_dict(torch.load(cond_m))
    # only for loading trainable embedding
    print "Loading from %s" % agg_e
    model.agg_type_embed_layer.load_state_dict(torch.load(agg_e))
    print "Loading from %s" % sel_e
    model.sel_type_embed_layer.load_state_dict(torch.load(sel_e))
    print "Loading from %s" % cond_e
    model.cond_type_embed_layer.load_state_dict(torch.load(cond_e))

    table_id = raw_input("Input the table id:")
    question = raw_input("Ask something:")
    tokenizer = TreebankWordTokenizer()

    data = list()
    tables = dict()

    while table_id != "quit" and question != "quit":

        print "generating queries for \"" + question + "\"..."
        entry = {"phase": 1,
                 "kg_entities": [],
                 "question_tok_space": [],
                 "table_id": table_id,
                 "question": question,
                 "query": "",
                 "query_tok": "",
                 "sql": {
                     "sel": 0,
                     "conds": [[0, 0, 0]],
                     "agg": 0
                 }}
        entry = json.dumps(entry)
        entry = json.loads(entry)
        with open("data/test_tok.tables.jsonl") as f:
            for line in f:
                table = json.loads(line.strip())
                tables[table["id"]] = table

        tokens = tokenizer.tokenize(question)
        entry[Keys.QUESTION_TOK] = tokens
        entry[Keys.QUESTION_TOK_ORG] = tokens

        # entry = add_question_tok_kgcol(entry)
        entry = group_words(entry, tables)

        res = [item[Keys.META_TOKS] for item in entry[Keys.META]]

        entry[Keys.QUESTION_TYPE_KGCOL] = res
        res = []
        for item in entry[Keys.META]:
            extra = []
            if item[Keys.TYPE] == Keys.TYPE_KG:
                extra.append(Keys.ENTITY)
            elif item[Keys.TYPE] == Keys.TYPE_HEADER:
                extra.append(Keys.COLUMN)
            try:
                res += [item[Keys.META_CLS] + extra]
            except:
                print item
                print entry
                print '------------------------------------'
        entry[Keys.QUESTION_TYPE_KGCOL_LIST] = res

        # add question_type_org_kgcol
        res = []
        for item in entry[Keys.META]:
            res += [item[Keys.META_SIM_CLS]] * len(item[Keys.META_TOKS])
        entry[Keys.QUESTION_TYPE_ORG_KGCOL] = res
        del entry[Keys.META]
        entry = group_words_col(entry, tables)
        print json.dumps(entry)
        # f.write(json.dumps(entry) + "\n")
        # parse question format
        data.append(entry)

        model.eval()
        perm = list(range(1))
        st = 0
        ed = 1
        TEST_ENTRY = (True, True, True)

        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, q_type, col_type, \
        raw_data = to_batch_seq(
            data, test_table_data, idxes=perm, st=st, ed=ed, db_content=1, ret_vis_data=True)

        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        query_gt, table_ids = to_batch_query(data, perm, st, ed)
        gt_sel_seq = [x[1] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num,
                              q_type, col_type, pred_entry=TEST_ENTRY)
        pred_queries = model.gen_query(score, q_seq, col_seq,
                                       raw_q_seq, raw_col_seq, pred_entry=TEST_ENTRY)
        print(json.dumps(pred_queries))
        table_id = raw_input("Input the table id:")
        question = raw_input("Ask something:")
