import os
from . import utils
import pandas as pd
import numpy as np


class PythonDRDriver:


    def __init__(self):
        pass

    def run(self, expression_mat, metadata_dataframe):

        meta_data = metadata_dataframe.copy()
        exp_mat = expression_mat.copy()

        special_char_dictionary = {'+' : 'specialplus','-' : 'specialminus', '.' : 'specialperiod' , '/':'specialslash','\\':'special_back_slash',')':'special_paren_backward',
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

        cond = meta_data['condName'].copy()
        prev = meta_data['prevCol'].copy()
        delt = meta_data['del.t'].copy()
        delTmin = self.delTmin
        delTmax = self.delTmax
        tau = self.tau
        prev.loc[delt > delTmax] = np.nan
        delt.loc[delt > delTmax] = np.nan
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

        des_mat=pd.DataFrame(None,index=exp_mat.index,columns=None)
        res_mat=pd.DataFrame(None,index=exp_mat.index,columns=None)
        steady = prev.isnull() & ~(cond.isin(prev))

        des_mat=pd.concat([des_mat, exp_mat[cond[steady]]], axis=1)
        res_mat=pd.concat([res_mat, exp_mat[cond[steady]]], axis=1)


        for i in list(np.where(~steady)[0]):
            following = list(np.where(prev.str.contains(cond[i])==True)[0])
            following_delt = list(delt[following])

            try:
                off = list(np.where(following_delt[0] < delTmin)[0])
            except:
                off = []

            while len(off)>0:
                off_fol = list(np.where(prev.str.contains(cond[following[off[0]]])==True)[0])
                off_fol_delt = list(delt[off_fol])
                following=following[:off[0]] + following[off[0]+1:] + off_fol
                following_delt = following_delt[:off[0]] + following_delt[off[0]+1:]+[float(off_fol_delt[0]) + float(following_delt[off[0]])]
                off = list(np.where(following_delt < delTmin)[0])

            n = len(following)
            cntr = 0

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



                des_tmp = np.concatenate((des_mat.values,exp_mat[cond[i]].values[:,np.newaxis]),axis=1)
                des_names = list(des_mat.columns)+[this_cond]
                des_mat=pd.DataFrame(des_tmp,index=des_mat.index,columns=des_names)
                interp_res = (float(tau)/float(following_delt[cntr])) * (exp_mat[cond[j]].astype('float64') - exp_mat[cond[i]].astype('float64')) + exp_mat[cond[i]].astype('float64')
                res_tmp = np.concatenate((res_mat.values,interp_res.values[:,np.newaxis]),axis=1)
                res_names = list(res_mat.columns)+[this_cond] #list(exp_mat[cond[i]].columns)
                res_mat=pd.DataFrame(res_tmp,index=res_mat.index,columns=res_names)

                cntr = cntr + 1

            # special case: nothing is following this condition within delT.min
            # and it is the first of a time series --- treat as steady state

            if n == 0 and prev.isnull()[i]:

                des_mat = pd.concat([des_mat, exp_mat[cond[i]]], axis=1)
                des_mat.rename(columns={des_mat.columns.values[len(des_mat.columns)-1]:cond[i]}, inplace=True)
                res_mat = pd.concat([res_mat, exp_mat[cond[i]]], axis=1)
                res_mat.rename(columns={res_mat.columns.values[len(res_mat.columns)-1]:cond[i]}, inplace=True)


        cols_des_mat=des_mat.columns.tolist()
        cols_res_mat=res_mat.columns.tolist()

        special_char_inv_map = {v: k for k, v in special_char_dictionary.iteritems()}
        for sch in special_char_inv_map.keys():
            cols_des_mat=[item.replace(sch, special_char_inv_map[sch]) for item in cols_des_mat]
            cols_res_mat=[item.replace(sch, special_char_inv_map[sch]) for item in cols_res_mat]

        des_mat.columns=cols_des_mat
        res_mat.columns=cols_res_mat

        return (des_mat, res_mat)
