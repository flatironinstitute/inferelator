import os
from . import utils
import pandas as pd
import numpy as np

class PythonDRDriver:

    #meta_file = "meta_data.csv"
    #exp_file = "exp_mat.csv"
    #script_file = "run_design_response.R"
    #response_file = "response.tsv"
    #design_file = "design.tsv"
    #delTmin = 0
    #delTmax = 110
    #tau = 45

    def __init__(self):
        #self.delT_min = 2
        #with regards to naming, replace "." within names with "_"
        # self.meta_data = pd.read_csv('/Users/tymorhamamsy/inferelator_ng/inferelator_ng/tests/artifacts/meta_data.csv',sep = ',',index_col=0, header = 0)
        # self.exp_mat = pd.read_csv('/Users/tymorhamamsy/inferelator_ng/inferelator_ng/tests/artifacts/exp_mat.csv', sep = ',', index_col=0, header = 0)
        #self.delTmin = 2
        #self.delT_max = 4
        #self.tau = 2
        pass

    def run(self, expression_mat, metadata_dataframe):

        #import pdb
        #pdb.set_trace()
        meta_data = metadata_dataframe.copy()
        exp_mat = expression_mat.copy()

        special_char_dictionary = {'+' : 'specialplus','-' : 'specialminus','/':'specialslash','\\':'special_back_slash',')':'special_paren_backward',
        '(':'special_paren_forward', ',':'special_comma', ':':'special_colon',';':'special_semicoloon','@':'special_at','=':'special_equal',
         '>':'special_great','<':'special_less','[':'special_left_bracket',']':'special_right_bracket',"%":'special_percent',"*":'special_star',
        '&':'special_ampersand','^':'special_arrow','?':'special_question','!':'special_exclamation','#':'special_hashtag',"{":'special_left_curly',
        '}':'special_right_curly','~':'special_tilde','`':'special_tildesib','$':'special_dollar','|':'special_vert_bar'}

        cols=exp_mat.columns.tolist()
        for ch in special_char_dictionary.keys():
            meta_data['condName']=meta_data['condName'].str.replace(ch,special_char_dictionary[ch])
            meta_data['prevCol']=meta_data['prevCol'].str.replace(ch,special_char_dictionary[ch])
            cols=[item.replace(ch,special_char_dictionary[ch]) for item in cols]
        exp_mat.columns=cols

        '''
        meta_data['condName']=meta_data['condName'].str.replace("+", "aaaaa")
        meta_data['prevCol']=meta_data['prevCol'].str.replace("+", "aaaaa")
        meta_data['condName']=meta_data['condName'].str.replace("-", "bbbbb")
        meta_data['prevCol']=meta_data['prevCol'].str.replace("-", "bbbbb")
        meta_data['condName']=meta_data['condName'].str.replace("/", "ccccc")
        meta_data['prevCol']=meta_data['prevCol'].str.replace("/", "ccccc")
        cols=exp_mat.columns.tolist()
        cols=[item.replace("+", "aaaaa") for item in cols]
        cols=[item.replace("-", "bbbbb") for item in cols]
        cols=[item.replace("/", "ccccc") for item in cols]
        exp_mat.columns=cols
        '''
        cond = meta_data['condName']
        prev = meta_data['prevCol']
        delt = meta_data['del.t']
        delTmin = self.delTmin
        delTmax = self.delTmax
        tau = self.tau
        #prev.loc[delt > self.delTmin] = 'NaN'
        #delt.loc[delt > self.delTmax] = 'NaN'
        prev.loc[delt > delTmax] = np.nan #'NaN'
        delt.loc[delt > delTmax] = np.nan #'NaN'
        #print prev
        not_in_mat=set(cond)-set(exp_mat)
        cond_dup = cond.duplicated()
        if len(not_in_mat) > 0:
            cond = cond.str.replace('[/+-]', '.')
            prev = cond.str.replace('[/+-]', '.')
            if cond_dup != cond.duplicated():
                raise ValueError('Tried to fix condition names in meta data so that they would match column names in expression matrix, but failed')

                # check if there are condition names missing in expression matrix
        not_in_mat=set(cond)-set(exp_mat)
        if len(not_in_mat) > 0:
            print not_in_mat
            raise ValueError('Error when creating design and response. The conditions printed above are in the meta data, but not in the expression matrix')

        #pd.matrix(list(exp_mat.index)
        des_mat=pd.DataFrame(None,index=exp_mat.index,columns=None)
        res_mat=pd.DataFrame(None,index=exp_mat.index,columns=None)

        steady = prev.isnull() & ~(cond.isin(prev))
        #print steady
        #print list(np.where(~steady)[0])

        des_mat=pd.concat([des_mat, exp_mat[cond[steady]]], axis=1)
        #print des_mat
        res_mat=pd.concat([res_mat, exp_mat[cond[steady]]], axis=1)
        #print(res_mat)

        for i in list(np.where(~steady)[0]):
            following = list(np.where(prev.str.contains(cond[i])==True)[0])
            following_delt = list(delt.loc[following])
            #print following,following_delt

            try:
                off = list(np.where(following_delt[0] < delTmin)[0])
            except:
                off = []
            #print off
            #print type(off)
            #print off
            while len(off)>0:
                off_fol = list(np.where(prev.str.contains(cond[following[off[0]]])==True)[0])
                #print off_fol
                off_fol_delt = list(delt.loc[off_fol]) #might want it to be off_fol[0]
                #print off_fol_delt

                #print following
                #print off
                #print [following[-off[0]]]
                #print following[:off[0]] + following[off[0]+1:]

                #print off_fol
                #following=[following[-off[0]]]+off_fol
                following=following[:off[0]] + following[off[0]+1:] + off_fol
                #print following
                #print following
                #following_delt = [following_delt[-off[0]]]+[off_fol_delt+following_delt[off[0]]]
                #following_delt = following_delt[:off[0]] + following_delt[off[0]+1:]+off_fol_delt+[following_delt[off[0]]]
                #off_fol_delt.append(following_delt[off[0]])
                #following_delt = following_delt[:off[0]] + following_delt[off[0]+1:]+ off_fol_delt
                following_delt = following_delt[:off[0]] + following_delt[off[0]+1:]+[float(off_fol_delt[0]) + float(following_delt[off[0]])]
                #print following_delt
                off = list(np.where(following_delt < delTmin)[0])
                #print off

            #print off_fol
            #print off_fol_delt
            #print following
            #print off
            n = len(following)
            #print n

            #print n
            cntr = 0
            #print following_delt
            #print type(following_delt)
            #print following_delt[0]
            for j in following:
                if n>1:
                    this_cond = "%s_dupl%02d" % (cond[i], cntr+1)
                    original_this_cond = this_cond
                    k = 1
                    while this_cond in res_mat.columns :
                        this_cond = original_this_cond + '.{}'.format(int(k))
                        k = k + 1
                else:
                    this_cond = cond[i]

                des_mat =  pd.concat([des_mat, exp_mat[cond[i]]], axis=1)
                des_mat.rename(columns={des_mat.columns.values[len(des_mat.columns)-1]:this_cond}, inplace=True)
                #print des_mat
                #print following_delt[cntr]

                #interp_res = int(tau/following_delt[cntr]) * (exp_mat[cond[j]] - exp_mat[cond[i]]) + exp_mat[cond[i]]
                #interp_res = tau/following_delt[cntr] * (exp_mat[cond[j]] - exp_mat[cond[i]]) + exp_mat[cond[i]]
                interp_res = (float(tau)/float(following_delt[cntr])) * (exp_mat[cond[j]].astype('float64') - exp_mat[cond[i]].astype('float64')) + exp_mat[cond[i]].astype('float64')
                #print interp_res
                res_mat = pd.concat([res_mat, interp_res], axis=1)
                #print res_mat
                res_mat.rename(columns={res_mat.columns.values[len(res_mat.columns)-1]:this_cond}, inplace=True)
                #print(res_mat)
                cntr = cntr + 1

            #print des_mat,res_mat
            # special case: nothing is following this condition within delT.min
            # and it is the first of a time series --- treat as steady state
            #print des_mat,res_mat

            if n == 0 and prev.isnull()[i]:
                print i
                #print exp_mat[cond[i]]
                #print des_mat
                des_mat = pd.concat([des_mat, exp_mat[cond[i]]], axis=1)
                #print des_mat
                des_mat.rename(columns={des_mat.columns.values[len(des_mat.columns)-1]:cond[i]}, inplace=True)
                res_mat = pd.concat([res_mat, exp_mat[cond[i]]], axis=1)
                res_mat.rename(columns={res_mat.columns.values[len(res_mat.columns)-1]:cond[i]}, inplace=True)

        resp_idx = np.repeat(np.matrix(range(1,len(res_mat.columns.values)+1)),len(exp_mat.index),axis=0)

        cols_des_mat=des_mat.columns.tolist()
        cols_res_mat=res_mat.columns.tolist()

        special_char_inv_map = {v: k for k, v in special_char_dictionary.iteritems()}
        #import pdb; pdb.set_trace()
        for sch in special_char_inv_map.keys():
            cols_des_mat=[item.replace(sch, special_char_inv_map[sch]) for item in cols_des_mat]
            cols_res_mat=[item.replace(sch, special_char_inv_map[sch]) for item in cols_res_mat]
        '''
        cols_des_mat=[item.replace("aaaaa", "+") for item in cols_des_mat]
        cols_res_mat=[item.replace("aaaaa", "+") for item in cols_des_mat]
        cols_des_mat=[item.replace("bbbbb", "-") for item in cols_des_mat]
        cols_res_mat=[item.replace("bbbbb", "-") for item in cols_des_mat]
        cols_des_mat=[item.replace("ccccc", "/") for item in cols_des_mat]
        cols_res_mat=[item.replace("ccccc", "/") for item in cols_des_mat]
        '''
        des_mat.columns=cols_des_mat
        res_mat.columns=cols_res_mat

        return (des_mat, res_mat)
