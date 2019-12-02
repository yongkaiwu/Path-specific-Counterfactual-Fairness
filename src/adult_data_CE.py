import os
from itertools import product

import cvxpy as cvx
import numpy as np
import pandas as pd

from model.loadxml import load_xml_to_cbn
from model.variable import Event

cwd = os.getcwd()


def tight_infer_with_partial_graph(y_val, s1_val, s0_val, oa, oe, om):
    partial_cbn = load_xml_to_cbn(partial_model)
    partial_cbn.build_joint_table()

    Age = partial_cbn.v['age']
    Edu = partial_cbn.v['education']
    Sex = partial_cbn.v['sex']
    Workclass = partial_cbn.v['workclass']
    Marital = partial_cbn.v['marital-status']
    Hours = partial_cbn.v['hours']
    Income = partial_cbn.v['income']

    if s1_val == s0_val:
        # there is no difference when active value = reference value
        return 0.00, 0.00
    else:
        # define variable for P(r)
        PR = cvx.Variable(Marital.domain_size ** 8)

        # define ell functions
        g = {}
        for v in {Marital}:
            v_index = v.index
            v_domain_size = v.domain_size
            parents_index = partial_cbn.index_graph.pred[v_index].keys()
            parents_domain_size = np.prod([partial_cbn.v[i].domain_size for i in parents_index])
            g[v_index] = list(product(range(v_domain_size), repeat=int(parents_domain_size)))

        # format
        # [(), (), ()]
        # r corresponds to the tuple
        # parents corresponds to the location of the tuple

        # assert the response function. (t function of Pearl, I function in our paper)
        def Indicator(obs, parents, response):
            # sort the parents by id
            par_key = parents.keys()
            # map the value to index
            par_index = 0
            for k in par_key:
                par_index = par_index * partial_cbn.v[k].domain_size + parents.dict[k]

            return 1 if obs.first_value() == g[obs.first_key()][response][par_index] else 0

        # build the object function
        weights = np.zeros(shape=[Marital.domain_size ** 8])

        for rm in range(Marital.domain_size ** 8):
            # assert r -> o to obtain the conditional individuals
            product_i = 1
            for (obs, parents, response) in [(Event({Marital: om}), Event({Sex: s0_val, Age: oa, Edu: oe}), rm)]:
                product_i *= Indicator(obs, parents, response)

            if product_i == 1:
                # if ALL I()= 1, then continue the counterfactual inference
                # the first term for pse
                sum_identity = 0.0
                for m1, w, h in product(Marital.domains.get_all(), Workclass.domains.get_all(), Hours.domains.get_all()):
                    product_i = partial_cbn.get_prob(Event({Sex: s0_val}), Event({})) * \
                                partial_cbn.get_prob(Event({Age: oa}), Event({})) * \
                                partial_cbn.get_prob(Event({Edu: oe}), Event({Age: oa})) * \
                                Indicator(Event({Marital: m1}), Event({Sex: s1_val, Age: oa, Edu: oe}), rm) * \
                                partial_cbn.get_prob(Event({Workclass: w}), Event({Age: oa, Edu: oe, Marital: m1})) * \
                                partial_cbn.get_prob(Event({Hours: h}), Event({Workclass: w, Edu: oe, Marital: m1, Age: oa, Sex: s1_val})) * \
                                partial_cbn.get_prob(Event({Income: y_val}), Event({Sex: s1_val, Edu: oe, Workclass: w, Marital: m1, Hours: h, Age: oa}))

                    sum_identity += product_i

                weights[rm] += sum_identity

                # the second term for pse
                sum_identity = 0.0
                for m0, w, h in product(Marital.domains.get_all(), Workclass.domains.get_all(), Hours.domains.get_all()):
                    product_i = partial_cbn.get_prob(Event({Sex: s0_val}), Event({})) * \
                                partial_cbn.get_prob(Event({Age: oa}), Event({})) * \
                                partial_cbn.get_prob(Event({Edu: oe}), Event({Age: oa})) * \
                                Indicator(Event({Marital: m0}), Event({Sex: s0_val, Age: oa, Edu: oe}), rm) * \
                                partial_cbn.get_prob(Event({Workclass: w}), Event({Age: oa, Edu: oe, Marital: m0})) * \
                                partial_cbn.get_prob(Event({Hours: h}), Event({Workclass: w, Edu: oe, Marital: m0, Age: oa, Sex: s0_val})) * \
                                partial_cbn.get_prob(Event({Income: y_val}), Event({Sex: s0_val, Edu: oe, Workclass: w, Marital: m0, Hours: h, Age: oa}))

                    sum_identity += product_i

                weights[rm] -= sum_identity

        # build the objective function
        objective = weights.reshape(1, -1) @ PR / partial_cbn.get_prob(Event({Sex: s0_val, Age: oa, Edu: oe, Marital: om}))

        ############################
        ### to build the constraints
        ############################

        ### the inferred model is consistent with the observational distribution
        A_mat = np.zeros((Age.domain_size, Edu.domain_size, Marital.domain_size, Sex.domain_size,
                          Marital.domain_size ** 8))
        b_vex = np.zeros((Age.domain_size, Edu.domain_size, Marital.domain_size, Sex.domain_size))

        # assert r -> v
        for a, e, m, s in product(Age.domains.get_all(),
                                  Edu.domains.get_all(),
                                  Marital.domains.get_all(),
                                  Sex.domains.get_all()):
            # calculate the probability of observation
            b_vex[a.index, e.index, m.index, s.index] = partial_cbn.get_prob(Event({Age: a, Edu: e, Marital: m, Sex: s}))
            # sum of P(r)
            for rm in range(Marital.domain_size ** 8):
                product_i = partial_cbn.get_prob(Event({Sex: s}), Event({})) * \
                            partial_cbn.get_prob(Event({Age: a}), Event({})) * \
                            partial_cbn.get_prob(Event({Edu: e}), Event({Age: a})) * \
                            Indicator(Event({Marital: m}), Event({Sex: s, Age: a, Edu: e}), rm)
                A_mat[a.index, e.index, m.index, s.index, rm] = product_i

        # flatten the matrix and vector
        A_mat = A_mat.reshape(-1, Marital.domain_size ** 8)
        b_vex = b_vex.reshape(-1, 1)

        ### the probability <= 1
        C_mat = np.identity(Marital.domain_size ** 8)
        d_vec = np.ones(Marital.domain_size ** 8)

        ### the probability is positive
        E_mat = np.identity(Marital.domain_size ** 8)
        f_vec = np.zeros(Marital.domain_size ** 8)

        constraints = [
            A_mat @ PR == b_vex,
            C_mat @ PR <= d_vec,
            E_mat @ PR >= f_vec
        ]

        # minimize the causal effect
        problem = cvx.Problem(cvx.Minimize(objective), constraints)
        problem.solve()

        # print('tight lower effect: %f' % (problem.value))
        lower = problem.value

        # maximize the causal effect
        problem = cvx.Problem(cvx.Maximize(objective), constraints)
        problem.solve()

        # print('tight upper effect: %f' % (problem.value))
        upper = problem.value

        return upper, lower


def loosely_infer_with_partial_graph(y_val, s1_val, s0_val, oa, oe, om):
    partial_cbn = load_xml_to_cbn(partial_model)
    partial_cbn.build_joint_table()

    Age = partial_cbn.v['age']
    Edu = partial_cbn.v['education']
    Sex = partial_cbn.v['sex']
    Workclass = partial_cbn.v['workclass']
    Marital = partial_cbn.v['marital-status']
    Hours = partial_cbn.v['hours']
    Income = partial_cbn.v['income']

    def max_w_h():
        p_max = 0.0
        p_min = 1.0
        for m in Marital.domains.get_all():
            p_m = partial_cbn.get_prob(Event({Income: y_val}), Event({Sex: s1_val, Age: oa, Edu: oe, Marital: m}))
            p_max = max(p_m, p_max)
            p_min = min(p_m, p_min)
        return p_max, p_min

    p1_upper, p1_lower = max_w_h()

    p2 = partial_cbn.get_prob(Event({Income: y_val}), Event({Sex: s0_val, Age: oa, Edu: oe, Marital: om}))
    return p1_upper - p2, p1_lower - p2


tau = 0.05
spos = 1
sneg = 0

data_dir = '../data/adult/'
temp_dir = '../temp/adult/'
partial_model = data_dir + 'adult.xml'

if __name__ == '__main__':

    df = pd.DataFrame(data=np.zeros((8, 4)))

    for i, (oa, oe, om) in enumerate(product([0, 1], [0, 1], [0, 1])):
        print('*' * 8, '\noa, oe, om =', (oa, oe, om))

        p_u, p_l = loosely_infer_with_partial_graph(y_val=1, s1_val=spos, s0_val=sneg, oa=oa, oe=oe, om=om)
        print('loose effect: %0.4f, %0.4f' % (p_l, p_u))
        df.loc[i, :1] = p_l, p_u

        p_u, p_l = tight_infer_with_partial_graph(y_val=1, s1_val=spos, s0_val=sneg, oa=oa, oe=oe, om=om)
        print('tight effect: %0.4f, %0.4f' % (p_l, p_u))
        df.loc[i, 2:] = p_l, p_u

    df.columns = ['lb', 'ub', 'lb', 'ub']
    print(df.round(4))
