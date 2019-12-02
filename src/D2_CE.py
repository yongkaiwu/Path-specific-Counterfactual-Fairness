import os
from itertools import product

import cvxpy as cvx
import numpy as np

from model.loadxml import load_xml_to_cbn
from model.variable import Event
from preprocessing import preprocess

cwd = os.getcwd()


def infer_with_complete_graph(y_val, s1_val, s0_val, ow, oa):
    complete_cbn = load_xml_to_cbn(complete_model)

    US = complete_cbn.v['US']
    UW = complete_cbn.v['UW']
    UA = complete_cbn.v['UA']
    UY = complete_cbn.v['UY']

    S = complete_cbn.v['S']
    W = complete_cbn.v['W']
    A = complete_cbn.v['A']
    Y_hat = complete_cbn.v['Y']

    complete_cbn.build_joint_table()
    p1, p2 = 0.0, 0.0
    for us, uw, ua, uy in product(US.domains.get_all(), UW.domains.get_all(), UA.domains.get_all(), UY.domains.get_all()):
        # compute p(u|z, s)
        ps = complete_cbn.get_prob(Event({US: us, UW: uw, UA: ua}), Event({S: s0_val, W: ow, A: oa}))
        # print(ps)
        if ps == 0.00000:
            continue
        for w1, a1 in product(W.domains.get_all(), A.domains.get_all()):
            p1 += complete_cbn.get_prob(Event({W: w1}), Event({UW: uw, S: s1_val})) * \
                  complete_cbn.get_prob(Event({A: a1}), Event({UA: ua, W: w1})) * \
                  complete_cbn.get_prob(Event({Y_hat: y_val}), Event({UY: uy, S: s1_val, W: w1, A: a1})) * \
                  complete_cbn.get_prob(Event({UY: uy}), Event({})) * ps

        for w0, a0 in product(W.domains.get_all(), A.domains.get_all()):
            p2 += complete_cbn.get_prob(Event({W: w0}), Event({UW: uw, S: s0_val})) * \
                  complete_cbn.get_prob(Event({A: a0}), Event({UA: ua, W: w0})) * \
                  complete_cbn.get_prob(Event({Y_hat: y_val}), Event({S: s0_val, W: w0, A: a0})) * \
                  complete_cbn.get_prob(Event({UY: uy}), Event({})) * ps

    assert abs(p2 - complete_cbn.get_prob(Event({Y_hat: y_val}), Event({S: s0_val, W: ow, A: oa}))) < 0.001
    # print(p1, p2)
    return p1, p2


def tight_infer_with_partial_graph(y_val, s1_val, s0_val, ow, oa):
    partial_cbn = load_xml_to_cbn(partial_model)
    partial_cbn.build_joint_table()

    S = partial_cbn.v['S']
    W = partial_cbn.v['W']
    A = partial_cbn.v['A']
    Y_hat = partial_cbn.v['Y']

    if s1_val == s0_val:
        # there is no difference when active value = reference value
        return 0.00, 0.00
    else:
        # define variable for P(r)
        PR = cvx.Variable(W.domain_size ** S.domain_size * A.domain_size ** W.domain_size)

        # define ell functions
        g = {}
        for v in {S, W, A, Y_hat}:
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
        weights = np.zeros(shape=[W.domain_size ** S.domain_size, A.domain_size ** W.domain_size])

        for rw, ra in product(range(W.domain_size ** S.domain_size), range(A.domain_size ** W.domain_size)):
            # assert r -> o to obtain the conditional individuals
            product_i = 1
            for (obs, parents, response) in [(Event({W: ow}), Event({S: s0_val}), rw),
                                             (Event({A: oa}), Event({W: ow}), ra)]:
                product_i *= Indicator(obs, parents, response)

            if product_i == 1:
                # if ALL I()= 1, then continue the counterfactual inference
                # the first term for pse
                sum_identity = 0.0
                for w1, a1 in product(W.domains.get_all(), A.domains.get_all()):
                    product_i = partial_cbn.get_prob(Event({S: s0_val}), Event({})) * \
                                Indicator(Event({W: w1}), Event({S: s1_val}), rw) * \
                                Indicator(Event({A: a1}), Event({W: w1}), ra) * \
                                partial_cbn.get_prob(Event({Y_hat: y_val}), Event({S: s1_val, W: w1, A: a1}))

                    sum_identity += product_i

                weights[rw, ra] += sum_identity

                # the second term for pse
                sum_identity = 0.0
                for w0, a0 in product(W.domains.get_all(), A.domains.get_all()):
                    product_i = partial_cbn.get_prob(Event({S: s0_val}), Event({})) * \
                                Indicator(Event({W: w0}), Event({S: s0_val}), rw) * \
                                Indicator(Event({A: a0}), Event({W: w0}), ra) * \
                                partial_cbn.get_prob(Event({Y_hat: y_val}), Event({S: s0_val, W: w0, A: a0}))

                    sum_identity += product_i

                weights[rw, ra] -= sum_identity

        # build the objective function
        objective = weights.reshape(1, -1) @ PR / partial_cbn.get_prob(Event({S: s0_val, W: ow, A: oa}))

        ############################
        ### to build the constraints
        ############################

        ### the inferred model is consistent with the observational distribution
        A_mat = np.zeros((S.domain_size, W.domain_size, A.domain_size, Y_hat.domain_size,
                          W.domain_size ** S.domain_size, A.domain_size ** W.domain_size))
        b_vex = np.zeros((S.domain_size, W.domain_size, A.domain_size, Y_hat.domain_size))

        # assert r -> v
        for s, w, a, y in product(S.domains.get_all(),
                                  W.domains.get_all(),
                                  A.domains.get_all(),
                                  Y_hat.domains.get_all()):
            # calculate the probability of observation
            b_vex[s.index, w.index, a.index, y.index] = partial_cbn.get_prob(Event({S: s, Y_hat: y, W: w, A: a}))
            # sum of P(r)
            for rw, ra in product(range(W.domain_size ** S.domain_size),
                                  range(A.domain_size ** W.domain_size)):
                product_i = partial_cbn.get_prob(Event({S: s}), Event({})) * \
                            partial_cbn.get_prob(Event({Y_hat: y}), Event({S: s, W: w, A: a})) * \
                            Indicator(Event({W: w}), Event({S: s}), rw) * \
                            Indicator(Event({A: a}), Event({W: w}), ra)

                A_mat[s.index, w.index, a.index, y.index, rw, ra] = product_i

        # flatten the matrix and vector
        A_mat = A_mat.reshape(S.domain_size * W.domain_size * A.domain_size * Y_hat.domain_size,
                              W.domain_size ** S.domain_size * A.domain_size ** W.domain_size)
        b_vex = b_vex.reshape(-1, 1)

        ### the probability <= 1
        C_mat = np.identity(W.domain_size ** S.domain_size * A.domain_size ** W.domain_size)
        d_vec = np.ones(W.domain_size ** S.domain_size * A.domain_size ** W.domain_size)

        ### the probability is positive
        E_mat = np.identity(W.domain_size ** S.domain_size * A.domain_size ** W.domain_size)
        f_vec = np.zeros(W.domain_size ** S.domain_size * A.domain_size ** W.domain_size)

        constraints = [
            A_mat @ PR == b_vex,
            # C_mat @ PR == d_vec,
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


def loosely_infer_with_partial_graph(y_val, s1_val, s0_val, ow, oa):
    partial_cbn = load_xml_to_cbn(partial_model)
    partial_cbn.build_joint_table()

    S = partial_cbn.v['S']
    W = partial_cbn.v['W']
    A = partial_cbn.v['A']
    Y_hat = partial_cbn.v['Y']

    def max_w_a():
        p_max = 0.0
        p_min = 1.0
        for w, a in product(W.domains.get_all(), A.domains.get_all()):
            p = partial_cbn.get_prob(Event({Y_hat: y_val}), Event({S: s1_val, W: w, A: a}))
            if p < p_min:
                p_min = p
            if p > p_max:
                p_max = p
        return p_max, p_min

    p1_upper, p1_lower = max_w_a()
    p2 = partial_cbn.get_prob(Event({Y_hat: y_val}), Event({S: s0_val, W: ow, A: oa}))

    return p1_upper - p2, p1_lower - p2


if __name__ == '__main__':
    tau = 0.05
    spos = 0
    sneg = 1

    data_dir = '../data/D2/'
    temp_dir = '../temp/D2/'
    complete_model = data_dir + 'complete_model.xml'
    partial_model = data_dir + 'observed_model.xml'

    preprocess(data_dir, temp_dir, seed=1)

    for i, (ow, oa) in enumerate(product([0, 1], [0, 1])):
        print('*' * 8, '\now, oa =', (ow, oa))

        p1, p2 = infer_with_complete_graph(y_val=1, s1_val=spos, s0_val=sneg, ow=ow, oa=oa)
        print('true effect: %0.4f' % (p1 - p2))

        p_u, p_l = loosely_infer_with_partial_graph(y_val=1, s1_val=spos, s0_val=sneg, ow=ow, oa=oa)
        print('loose effect: %0.4f, %0.4f' % (p_l, p_u))

        p_u, p_l = tight_infer_with_partial_graph(y_val=1, s1_val=spos, s0_val=sneg, ow=ow, oa=oa)
        print('tight effect: %0.4f, %0.4f' % (p_l, p_u))
