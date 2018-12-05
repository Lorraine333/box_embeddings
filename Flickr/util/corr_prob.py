from util import cube_exp_prob
import tensorflow as tf
import numpy as np


def correlation(logx, logy, logxy):
    logx = clip_log_prob(logx)
    logy = clip_log_prob(logy)
    logxy = clip_log_prob(logxy)

    lognume1 = logxy
    lognume2 = logx+logy
    logdomi = 0.5*(logx+logy+cube_exp_prob.log1mexp(-logx)+cube_exp_prob.log1mexp(-logy))
    corr = tf.exp(lognume1-logdomi)-tf.exp(lognume2-logdomi)

    return corr

def clip_log_prob(logx, min_v=-10, max_v=-0.00001):
    return tf.clip_by_value(logx, min_v, max_v)

def clip_prob(x, min_v = 0.00005, max_v = 0.99999):
    return tf.clip_by_value(x, min_v, max_v)

def lambda_corr_loss(join_min, join_max, meet_min, meet_max, t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed, x_labels, y_labels, xy_labels, not_have_meet):
    x_predicted = cube_exp_prob.probability(t1_min_embed, t1_max_embed)
    y_predicted = cube_exp_prob.probability(t2_min_embed, t2_max_embed)
    joint_predicted = cube_exp_prob.test_joint_probability_log(join_min, join_max, meet_min, meet_max, t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed, not_have_meet)
    pred_corr = correlation(x_predicted, y_predicted, joint_predicted)
    gold_corr = correlation(tf.log(clip_prob(x_labels)), tf.log(clip_prob(y_labels)), tf.log(clip_prob(xy_labels)))
    return tf.abs((pred_corr-gold_corr))

def lambda_upper_bound(join_min, join_max, meet_min, meet_max, t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed, x_labels, y_labels, xy_labels, not_have_meet):
    joint_log = cube_exp_prob.batch_log_upper_bound(join_min, join_max, t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed)
    return joint_log

def np_log1mexp(x):
    result = np.zeros(x.shape)
    for i in range(len(x)):
        if x[i] <= np.log(2.0):
		    result[i] = np.log(-np.expm1(-x[i]))
        else:
		    result[i] = np.log1p(-np.exp(-x[i]))
    return result

def np_correlation(logx, logy, logxy):
    logx = np_clip_log_prob(logx)
    logy = np_clip_log_prob(logy)
    logxy = np_clip_log_prob(logxy)

    lognume1 = logxy
    lognume2 = logx+logy
    logdomi = 0.5*(logx+logy+np_log1mexp(-logx)+np_log1mexp(-logy))
    corr = np.exp(lognume1-logdomi)-np.exp(lognume2-logdomi)

    return corr

def np_clip_log_prob(logx, min_v=-10, max_v=-0.00001):
    return np.clip(logx, min_v, max_v)

def np_clip_prob(x, min_v = 0.00005, max_v = 0.99999):
    return np.clip(x, min_v, max_v)

def corr_loss(x_predicted, y_predicted, joint_predicted, x_labels, y_labels, xy_labels):
    pred_corr = correlation(x_predicted, y_predicted, joint_predicted)
    gold_corr = correlation(tf.log(clip_prob(x_labels)), tf.log(clip_prob(y_labels)), tf.log(clip_prob(xy_labels)))
    result = tf.abs(pred_corr-gold_corr)
    return result