# This code is adapted from the implementation of MT-SGD (https://github.com/VietHoang1512/MT-SGD).


import altair as alt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
from tqdm.auto import tqdm
import torch.nn.functional as F


alt.data_transformers.enable("default", max_rows=None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import altair as alt
from vega_datasets import data


class MinNormSolver:
    MAX_ITER = 250
    STOP_CRIT = 1e-5

    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        if v1v2 >= v1v1:
            # Case: Fig 1, third column
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            # Case: Fig 1, first column
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        # Case: Fig 1, second column
        gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
        cost = v2v2 + gamma * (v1v2 - v2v2)
        return gamma, cost

    def _min_norm_2d(vecs, dps, decomposition):

        if decomposition:
            dmin = 5e8
            for i in range(len(vecs)):
                for j in range(i + 1, len(vecs)):
                    if (i, j) not in dps:
                        dps[(i, j)] = 0.0
                        for k in range(len(vecs[i])):
                            dps[(i, j)] += torch.mul(vecs[i][k], vecs[j][k]).sum()
                        dps[(j, i)] = dps[(i, j)]
                    if (i, i) not in dps:
                        dps[(i, i)] = 0.0
                        for k in range(len(vecs[i])):
                            dps[(i, i)] += torch.mul(vecs[i][k], vecs[i][k]).sum()
                    if (j, j) not in dps:
                        dps[(j, j)] = 0.0
                        for k in range(len(vecs[i])):
                            dps[(j, j)] += torch.mul(vecs[j][k], vecs[j][k]).sum()
                    c, d = MinNormSolver._min_norm_element_from2(dps[(i, i)], dps[(i, j)], dps[(j, j)])
                    if d < dmin:
                        dmin = d
                        sol = [(i, j), c, d]
        else:
            dmin = 5e8
            for i in range(len(vecs)):
                for j in range(i + 1, len(vecs)):
                    if (i, j) not in dps:
                        dps[(i, j)] = vecs[i][j]
                        dps[(j, i)] = dps[(i, j)]
                    if (i, i) not in dps:
                        dps[(i, i)] = vecs[i][i]
                    if (j, j) not in dps:
                        dps[(j, j)] = vecs[j][j]
                    c, d = MinNormSolver._min_norm_element_from2(dps[(i, i)], dps[(i, j)], dps[(j, j)])
                    if d < dmin:
                        dmin = d
                        sol = [(i, j), c, d]
            # print("dps", dps)
        return sol, dps

    def _projection2simplex(y):
        m = len(y)
        # print("torch.sort(y)", torch.sort(y)[0])
        sorted_y = torch.flip(torch.sort(y)[0], dims=[0])
        tmpsum = 0.0
        tmax_f = (y.sum() - 1.0) / m
        for i in range(m - 1):
            tmpsum += sorted_y[i]
            tmax = (tmpsum - 1) / (i + 1.0)
            if tmax > sorted_y[i + 1]:
                tmax_f = tmax
                break
        return torch.max(y - tmax_f, torch.zeros(y.shape))#.cuda())

    def _next_point(cur_val, grad, n):
        proj_grad = grad - (torch.sum(grad) / n)
        tm1 = -1.0 * cur_val[proj_grad < 0] / proj_grad[proj_grad < 0]
        tm2 = (1.0 - cur_val[proj_grad > 0]) / (proj_grad[proj_grad > 0])
        t = 1

        if len(tm1[tm1 > 1e-7]) > 0:
            t = (tm1[tm1 > 1e-7]).min()
        if len(tm2[tm2 > 1e-7]) > 0:
            t = min(t, (tm2[tm2 > 1e-7]).min())

        next_point = proj_grad * t + cur_val
        next_point = MinNormSolver._projection2simplex(next_point)
        return next_point

    def find_min_norm_element(vecs, decomposition):

        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps, decomposition)

        n = len(vecs)
        sol_vec = torch.zeros(n)#.cuda()
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec, init_sol[2]

        iter_count = 0

        grad_mat = torch.zeros((n, n))#.cuda()
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:

            grad_dir = -1.0 * torch.mm(grad_mat, sol_vec.view(-1, 1)).view(-1)
            new_point = MinNormSolver._next_point(sol_vec, grad_dir, n)
            # Re-compute the inner products for line search
            v1v1 = 0.0
            v1v2 = 0.0
            v2v2 = 0.0
            for i in range(n):
                for j in range(n):
                    v1v1 += sol_vec[i] * sol_vec[j] * dps[(i, j)]
                    v1v2 += sol_vec[i] * new_point[j] * dps[(i, j)]
                    v2v2 += new_point[i] * new_point[j] * dps[(i, j)]
            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)

            new_sol_vec = nc * sol_vec + (1 - nc) * new_point
            change = new_sol_vec - sol_vec
            # print("Change: ", change)
            try:
                if change.pow(2).sum() < MinNormSolver.STOP_CRIT:
                    return sol_vec, nd
            except Exception as e:
                print(e)
                print("Change: ", change)
                # return sol_vec, nd
            sol_vec = new_sol_vec
        return sol_vec, nd

def gradient_normalizers(grads, losses, normalization_type):
	gn = {}

	if normalization_type == "l2":
		for t in range(len(grads)):
			gn[t] = np.sqrt(np.sum([gr.pow(2).sum().data.cpu() for gr in grads[t]]))

	elif normalization_type == "loss":
		for t in range(len(grads)):
			gn[t] = losses[t]


	elif normalization_type == "loss+":
		for t in range(len(grads)):
			gn[t] = losses[t] * np.sqrt(np.sum([gr.pow(2).sum().data.cpu() for gr in grads[t]]))

	elif normalization_type == "none":
		for t in range(len(grads)):
			gn[t] = 1.0

	else:
		print("ERROR: Invalid Normalization Type")
	return gn



class MoG(torch.distributions.Distribution):
	def __init__(self, pi, loc, covariance_matrix):
		#super(MoG, self).__init__(torch.Size([]), torch.Size([loc.size(-1)]))#
		#super(MoG, self).__init__()
		
		self.num_components = loc.size(0)
		self.loc = loc
		#self.covariance_matrix = covariance_matrix
		self.pi = pi
		self.cov = covariance_matrix
		#self.dists =[
		#	torch.distributions.MultivariateNormal(mu, covariance_matrix=sigma)
		#	for mu, sigma in zip(loc, covariance_matrix)
		#]
		self.dists =[]
		for mu, sigma in zip(self.loc, self.cov):
			self.dists.append(torch.distributions.MultivariateNormal(mu, covariance_matrix=sigma))
		
		

	@property
	def arg_constraints(self):
		return self.dists[0].arg_constraints

	@property
	def support(self):
		return self.dists[0].support

	@property
	def has_rsample(self):
		return False

	def log_prob(self, value):
		res = torch.cat(
			[
				(torch.log(torch.tensor(self.pi[i])) + self.dists[i].log_prob(value)).unsqueeze(-1)
				for i in range(len(self.dists))
			],
			dim=-1
			).logsumexp(dim=-1)
		return res
	def enumerate_support(self):
		return self.dists[0].enumerate_support()

class RBF(torch.nn.Module):
	def __init__(self, sigma =None):
		super(RBF, self).__init__()
		self.sigma = sigma
	def forward(self, X, Y, scale):
		XX = X.matmul(X.t())
		YY = Y.matmul(Y.t())
		XY = X.matmul(Y.t())
		dnorm2 = XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0) - 2 * XY
		if self.sigma is None:
			np_dnorm2 = dnorm2.detach().cpu().numpy()
			h = np.median(np_dnorm2)/(2* np.log(X.size(0)+1))
			sigma = np.sqrt(h).item()
		else:
			sigma = self.sigma
		sigma = sigma * scale
		gamma = 1.0/ (1e-8 + 2 * sigma **2)
		K_XY = (-gamma * dnorm2).exp()
		return K_XY


K = RBF()
class H(nn.Module):
	def __init__(self, lr = 1e-3, input_dim=2):
		super().__init__()
		self.fc1_dim = input_dim
		self.fc2_dim = 10
		self.fc3_dim = 10
		self.fc4_dim = 1

		self.fc1 = nn.Linear(self.fc1_dim, self.fc2_dim)
		self.fc2 = nn.Linear(self.fc2_dim, self.fc3_dim)
		self.fc3 = nn.Linear(self.fc3_dim, self.fc4_dim)

		self.fc1.weight.data.normal_(0,0.1)
		self.fc2.weight.data.normal_(0,0.1)
		self.fc3.weight.data.normal_(0,0.1)

		self.lr = lr
		self.optimizer = torch.optim.Adam(self.parameters(), lr = self.lr, betas=(0.0, 0.99))
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, X):
		X = self.fc1(X)
		X = F.relu(X)
		#X = self.fc2(X)
		#X = F.relu(X)
		return F.relu(self.fc3(X))+ 0.00001



class MWGRAD:
	def __init__(self, n, P, K, optimizer, nornalize="none", lr = 3e-2):
		self.P = P
		self.K = K
		self.n_targets = len(P)
		self.optim = optimizer
		self.nornalize = nornalize
		self.sol = torch.ones(self.n_targets)/self.n_targets
		self.lr = lr
		self.n = n
		self.weights = torch.ones(n)/n
		

	def logK(self, X, weights, scale=1):
		K_XX = self.K(X, X, scale=1)
		ws = weights[None, :]
		repeat_ws = ws.repeat(K_XX.shape[0], 1)
		return torch.log(torch.sum(K_XX * repeat_ws, dim=1))


	def compute_neg_probs(self, X):
		values = []
		for index in range(self.n_targets):
			value = -torch.sum(self.P[index].log_prob(X) * self.weights)
			values.append(value.item())
		m = np.mean(values)
		return values, m


	def phi_svgd(self, X, index, weights, scale=1):
		X = X.detach().requires_grad_(True)
		log_prob = self.P[index].log_prob(X)
		score_func = autograd.grad(log_prob.sum(), X)[0] # n_particles x dimension
		K_XX = self.K(X, X.detach(), scale=1)
		w = weights[None, :].repeat(K_XX.shape[0], 1)
		wK_XX = K_XX * w.detach()
		grad_K = -autograd.grad(wK_XX.sum(), X)[0]
		phi1 = wK_XX.detach().matmul(score_func)#/ X.size(0)
		phi2 = grad_K #/ X.size(0)
		phi = phi1 + phi2
		return phi, score_func, log_prob
	def phi_network(self, X, index, scale=1):
		X = X.detach().requires_grad_(True)
		H_net = H(lr=1e-3, input_dim=X.shape[1])
		log_prob = self.P[index].log_prob(X)
		score_func = autograd.grad(log_prob.sum(), X)[0] # n_particles x dimension

		for i in range(10):
			Y = X.detach().clone()
			#Y.requires_grad = True
			eps = torch.normal(0, 1, size=(Y.shape[0], Y.shape[1])) 
			log_H_eps = torch.log(H_net(eps).mean())
			m_log_X = torch.log(H_net(Y)).mean()
			
			
			loss = log_H_eps - m_log_X #+ 0.5* torch.sum(Y*Y)
			H_net.optimizer.zero_grad()
			autograd.backward(loss)
			H_net.optimizer.step()

		
		X.requires_grad = True
		output = torch.log(H_net(X))
		grad = autograd.grad(outputs =output, inputs = X, grad_outputs=torch.ones_like(output), retain_graph=True)[0]
		phi = -grad + 0.2*X + score_func 
		return phi, phi, log_prob







	def phi_blob(self, X, index, weights, scale=1):
		X = X.detach().requires_grad_(True)
		log_prob = self.P[index].log_prob(X)
		score_func = autograd.grad(log_prob.sum(), X)[0]
		K_XX = self.K(X, X.detach(), scale=1)
		w = weights[None, :].repeat(K_XX.shape[0], 1)
		wK_XX = K_XX * w.detach()
		grad_K = autograd.grad(wK_XX.sum(), X)[0]
		wK_X = 1.0/wK_XX.sum(dim=1)
		wK_X = wK_X[:, None]
		hi = grad_K* wK_X.repeat(1, grad_K.shape[1])
		
		phi = score_func - hi
		return phi, phi, log_prob

	def phi_lavengin(self, X, index, scale=1):
		X = X.detach().requires_grad_(True)
		log_prob = self.P[index].log_prob(X)
		score_func = autograd.grad(log_prob.sum(), X)[0]
		
		phi = score_func + np.sqrt(2/self.lr) * torch.randn_like(score_func)
		return phi, phi, log_prob



	def step(self, X, scale=1, mtype='svgd', updateweight=True):
		
		n_particles = X.shape[0]


		score_funcs=[]
		losses =[]
		phis = [[] for i in range(self.n_targets)]
		log_probs =[]

		for i in range(self.n_targets):
			self.optim.zero_grad()
			if mtype=='svgd':
				phi, score_func, log_prob = self.phi_svgd(X, i, self.weights, scale)
			elif mtype=='blob':
				phi, score_func, log_prob = self.phi_blob(X, i, self.weights, scale)
			elif mtype=='network':
				phi, score_func, log_prob = self.phi_network(X, i)
			elif mtype=='lavengin':
				phi, score_func, log_prob = self.phi_lavengin(X, i, scale)


			
			phis[i].append(Variable(phi.detach().clone(), requires_grad = False))
			score_func = torch.nn.functional.normalize(score_func, dim=0)
			score_funcs.append(Variable(score_func.detach().clone(), requires_grad=False))

			log_probs.append(log_prob)
		

		log_probs = torch.stack(log_probs, 0)
		logqz = self.logK(X, self.weights)

		g_X = logqz[None, :].repeat(log_probs.shape[0], 1) - log_probs
		sol = self.sol
		g_X_sol = torch.matmul(sol, g_X)
		exp_gx_sol = torch.exp(-g_X_sol)
		
		nom = torch.sum(exp_gx_sol[None, :].repeat(log_probs.shape[0], 1) * log_probs, dim=1)
		denom = torch.sum(exp_gx_sol)
		grad = nom/denom



		

		logp = torch.matmul(sol, log_probs)

		score_funcs = torch.stack(score_funcs, 0)
		#K_XX = self.K(X, X.detach(), scale)
		#grad_K = -autograd.grad(K_XX.sum(), X)[0]

		with torch.no_grad():
			Q = torch.zeros((self.n_targets, self.n_targets))
			if type=='svgd':
				K_XX = self.K(X, X, scale).detach()
			else:
				K_XX = torch.ones(X.shape[0],X.shape[0])

			for i in range(self.n_targets):
				for j in range(i, self.n_targets):
					Q[i][j] = torch.mul(K_XX, torch.matmul(score_funcs[i], score_funcs[j].T)).sum()
					Q[j][i] = Q[i][j]


			gn = gradient_normalizers(phis, losses, self.nornalize)
			for i in range(self.n_targets):
				for gr_i in range(len(phis[i])):
					phis[i][gr_i]/= gn[i]


			#sol, min_norm = MinNormSolver.find_min_norm_element(Q, decomposition=False)
			if updateweight is False:
				grad = 0.0
			#self.sol = self.sol - 0.001*(torch.matmul(Q, self.sol)- grad)
			#self.sol = MinNormSolver._projection2simplex(self.sol)
			scales =[]
			for i in range(self.n_targets):
				scales.append(float(self.sol[i]) * phis[i][0])
		X.grad = - torch.stack(scales).sum(dim=0)

		self.optim.step()
		

		delta_w = logqz - logp
		if updateweight:
			self.weights = self.weights * torch.exp(-0.001 * delta_w)
			self.weights = self.weights/torch.sum(self.weights)
		

def get_density_chart(P, d=7.0, step=0.1):
    xv, yv = torch.meshgrid([torch.arange(-d, d, step), torch.arange(-d, d, step)])
    pos_xy = torch.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)), dim=-1)
    p_xy = P.log_prob(pos_xy.to(device)).exp().unsqueeze(-1).cpu()
    #   print(p_xy)
    df = torch.cat([pos_xy, p_xy], dim=-1).numpy()
    df = pd.DataFrame(
        {
            "x": df[:, :, 0].ravel(),
            "y": df[:, :, 1].ravel(),
            "p": df[:, :, 2].ravel(),
        }
    )

    chart = (
        alt.Chart(df)
        .mark_point()
        .encode(
            x="x:Q",
            y="y:Q",
            color=alt.Color("p:Q", scale=alt.Scale(scheme="viridis")),
            tooltip=["x", "y", "p"],
        )
    )
    return chart


def get_density_charts(Ps, d=7.0, step=0.1):
    xv, yv = torch.meshgrid([torch.arange(-d, d, step), torch.arange(-d, d, step)])
    pos_xy = torch.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)), dim=-1)

    #   print([P.log_prob(pos_xy).exp().unsqueeze(-1).cpu() for P in Ps])
    p_xy = torch.stack([P.log_prob(pos_xy.to(device)).exp().unsqueeze(-1).cpu() for P in Ps]).mean(0)

    df = torch.cat([pos_xy, p_xy], dim=-1).numpy()
    df = pd.DataFrame(
        {
            "x": df[:, :, 0].ravel(),
            "y": df[:, :, 1].ravel(),
            "p": df[:, :, 2].ravel(),
        }
    )

    chart = (
        alt.Chart(df)
        .mark_point()
        .encode(
            x="x:Q",
            y="y:Q",
            color=alt.Color("p:Q", scale=alt.Scale(scheme="viridis")),
            tooltip=["x", "y", "p"],
        )
    )
    return chart


def get_particles_chart(X):
    df = pd.DataFrame(
        {
            "x": X[:, 0],
            "y": X[:, 1],
        }
    )
    chart = alt.Chart(df).mark_circle(color="red").encode(x="x:Q", y="y:Q")
    return chart



pi1=[0.2, 0.7, 0.1]
loc1=[[4.0, 4.0], [-2.0, 0.0], [2.0, 0.0]]
loc1 = torch.Tensor(loc1).to(device) 
loc1 += torch.randn_like(loc1) * 0.3
cov1 = torch.Tensor([0.5, 0.5]).diag().unsqueeze(0).repeat(3, 1, 1).to(device)


pi2=[0.2, 0.7, 0.1]
loc2=[[-4.0, 4.0], [-2.0, 0.0], [2.0, 0.0]]
loc2 = torch.Tensor(loc2).to(device) 
loc2 += torch.randn_like(loc1) * 0.3
cov2 = torch.Tensor([0.5, 0.5]).diag().unsqueeze(0).repeat(3, 1, 1).to(device)


pi3=[0.2, 0.7, 0.1]
loc3=[[-4.0, -4.0], [-2.0, 0.0], [2.0, 0.0]]
loc3 = torch.Tensor(loc3).to(device) 
loc3 += torch.randn_like(loc1) * 0.3
cov3 = torch.Tensor([0.5, 0.5]).diag().unsqueeze(0).repeat(3, 1, 1).to(device)

pi4=[0.2, 0.7, 0.1]
loc4=[[4.0, -4.0], [-2.0, 0.0], [2.0, 0.0]]
loc4 = torch.Tensor(loc4).to(device) 
loc3 += torch.randn_like(loc1) * 0.3
cov4 = torch.Tensor([0.5, 0.5]).diag().unsqueeze(0).repeat(3, 1, 1).to(device)



mog_1 = MoG(pi1, loc1, cov1)
mog_2 = MoG(pi2, loc2, cov2)
mog_3 = MoG(pi3, loc3, cov3)
mog_4 = MoG(pi4, loc4, cov4)



n = 50
X_init = (5 * torch.randn(n, 2)).to(device)
X_init.data = torch.clamp(X_init.data.clone(), min=-10 +1e-3, max=10 - 1e-3)


mog_chart = get_density_charts([mog_4, mog_3, mog_2, mog_1], d=10.0, step=0.1)


MTS_Xs = []
X = X_init.clone()
lr = 3e-2
svgd1 = MWGRAD(n, [mog_4, mog_3, mog_2, mog_1], K, optim.Adam([X], lr = lr), nornalize="none", lr = lr)
svgd2 = MWGRAD(n, [mog_4, mog_3, mog_2, mog_1], K, optim.Adam([X], lr = lr), nornalize="none", lr = lr)



updateweight = True
mtype='blob'
updateweight_ = False

for i in tqdm(range(1501), total=1501):
	if i > 600:
		updateweight_ = updateweight
	if i in [0, 500, 1000, 1500]:
		MTS_Xs.append(X.detach().clone().cpu().numpy())
	svgd1.step(X, mtype=mtype, updateweight=updateweight_)
	print(i, svgd1.compute_neg_probs(X), svgd1.sol)
	#print(X)
	#print(svgd1.weights)
result1 = svgd1.compute_neg_probs(X)



chart1 = mog_chart + get_particles_chart(MTS_Xs[0])
chart2 = mog_chart + get_particles_chart(MTS_Xs[1])
chart3 = mog_chart + get_particles_chart(MTS_Xs[2])
chart4 = mog_chart + get_particles_chart(MTS_Xs[3])

chart1.title = "Step 0"
chart2.title = "Step 500"
chart3.title = "Step 1000"
chart4.title = "Step 1500"

chart = (
    alt.hconcat(chart1, chart2, chart3, chart4, center=True)
    .configure_title(fontSize=20)
    .configure_axis(titleFontSize=16)
).show()



























