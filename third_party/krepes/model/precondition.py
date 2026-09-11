import torch
from torch.func import jvp, vjp

class BasePreconditioner:
    def __init__(self, model, params_to_precondition=None):

        self.model = model
        if params_to_precondition:
            self.params = params_to_precondition
        else:
            self.params = [p for p in model.parameters() if p.requires_grad]

    @torch.no_grad()
    def step(self, **kwargs):
        self._update_preconditioner(**kwargs)
    
        self._apply_preconditioner()

    def _update_preconditioner(self, **kwargs):
        raise NotImplementedError("Subclasses must implement _update_preconditioner.")

    def _apply_preconditioner(self):
        raise NotImplementedError("Subclasses must implement _apply_preconditioner.")


def conjugate_gradient_solver(A_mvp, b, max_iter=10, rtol=1e-6):

    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs_old = torch.dot(r, r)

    if rs_old < 1e-10:
        return x

    for i in range(max_iter):
        Ap = A_mvp(p)
        alpha = rs_old / torch.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = torch.dot(r, r)

        if torch.sqrt(rs_new) < rtol:
            break
        
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
        
    return x

class KmmPCGPreconditioner(BasePreconditioner):

    def __init__(self, model, params_to_precondition, reg_lambda=1e-3, cg_max_iter=10):
        super().__init__(model, params_to_precondition)
        if len(self.params)!= 1:
            raise ValueError("designed to precondition a single parameter matrix (model.A).")
    
        self.param_A = self.params[0]
        self.reg_lambda = reg_lambda
        self.cg_max_iter = cg_max_iter
        self.K_mm_reg = None

    def _update_preconditioner(self, K_mm=None, **kwargs):

        if K_mm is None:
            raise ValueError("K_mm matrix must be provided.")
        self.K_mm_reg = K_mm + self.reg_lambda * torch.eye(K_mm.size(0), device=K_mm.device)

    def _apply_preconditioner(self):
        if self.param_A.grad is None:
            return

        grad_A = self.param_A.grad
        M, k = grad_A.shape
        
        preconditioned_grad_A = torch.empty_like(grad_A)

        def A_mvp(v):
            return torch.matmul(self.K_mm_reg, v)

        for j in range(k):
            grad_col = grad_A[:, j]
            preconditioned_grad_col = conjugate_gradient_solver(
                A_mvp, grad_col, max_iter=self.cg_max_iter
            )
            preconditioned_grad_A[:, j] = preconditioned_grad_col
        self.param_A.grad.copy_(preconditioned_grad_A)


class BarlowTwinsGNHPreconditioner(BasePreconditioner):

    def __init__(self, model, params_to_precondition, loss_fn, damping=1e-3, cg_max_iter=10):
        super().__init__(model, params_to_precondition)
        if len(self.params)!= 1:
            raise ValueError("designed for a single matrix parameter (model.A).")
        
        self.param_A = self.params[0]
        self.loss_fn = loss_fn
        self.damping = damping
        self.cg_max_iter = cg_max_iter
        self.K_A = None
        self.K_B = None
        self.n_samples = None
        self.lambda_reg = None

    def _update_preconditioner(self, K_A=None, K_B=None, config=None, **kwargs):

        if K_A is None or K_B is None or config is None:
            raise ValueError("K_A, K_B, and config must be provided.")
        
        self.K_A = K_A
        self.K_B = K_B
        self.n_samples = K_A.shape[0]
        self.lambda_reg = torch.tensor(config.lambda_reg, device=K_A.device)

    def _compute_residuals(self, A_matrix):

 
        Z_A = (self.K_A @ A_matrix)
        Z_B = (self.K_B @ A_matrix)
        
        Z_A_c = Z_A - Z_A.mean(dim=0, keepdim=True)
        Z_B_c = Z_B - Z_B.mean(dim=0, keepdim=True)
        
        Z_A_norm = Z_A_c / (torch.norm(Z_A_c, dim=0, keepdim=True) + 1e-6)
        Z_B_norm = Z_B_c / (torch.norm(Z_B_c, dim=0, keepdim=True) + 1e-6)
        
  
        C = (Z_A_norm.T @ Z_B_norm) / self.n_samples
        k = A_matrix.shape[1]

        diag_res = torch.diag(C) - 1.0
        
        off_diag_mask = ~torch.eye(k, device=C.device, dtype=torch.bool)
        off_diag_res = C[off_diag_mask] * torch.sqrt(self.lambda_reg)
        
        return torch.cat([diag_res, off_diag_res])

    def _apply_preconditioner(self):

        if self.param_A.grad is None:
            return

        current_A = self.param_A.data
        grad_A = self.param_A.grad
        
        def h_gn_mvp(d_vec):
            d_matrix = d_vec.reshape(current_A.shape)
            _, jvp_result = jvp(self._compute_residuals, (current_A,), (d_matrix,))
            vjp_result = vjp(self._compute_residuals, current_A)[1](jvp_result)
            
            hvp_flat = vjp_result[0].flatten()
            damped_hvp = hvp_flat + self.damping * d_vec
            return damped_hvp

        preconditioned_grad_flat = conjugate_gradient_solver(
            h_gn_mvp, grad_A.flatten(), max_iter=self.cg_max_iter
        )
        
        self.param_A.grad.copy_(preconditioned_grad_flat.reshape(grad_A.shape))

class SimclrGNHPreconditioner(BasePreconditioner):

    def __init__(self, model, params_to_precondition, temperature, damping=1e-3, cg_max_iter=10):
        super().__init__(model, params_to_precondition)
        if len(self.params)!= 1:
            raise ValueError("designed for a single matrix parameter (model.A).")
        
        self.param_A = self.params[0]
        self.tau = temperature
        self.damping = damping
        self.cg_max_iter = cg_max_iter
        self.K_nm = None
        self.K_nm_pos = None

    def _update_preconditioner(self, K_A=None, K_B=None, **kwargs):

        if K_A is None or K_B is None:
            raise ValueError("K_A (K_nm) and K_B (K_nm_pos) must be provided.")
        
        self.K_nm = K_A
        self.K_nm_pos = K_B

    def _compute_residuals(self, A_matrix):

        f_x = torch.matmul(self.K_nm, A_matrix)
        f_x_pos = torch.matmul(self.K_nm_pos, A_matrix)
        
        f_x = torch.nn.functional.normalize(f_x, p=2, dim=1)
        f_x_pos = torch.nn.functional.normalize(f_x_pos, p=2, dim=1)
        
        all_embeddings = torch.cat([f_x, f_x_pos], dim=0)
        logits = (all_embeddings @ all_embeddings.T) / self.tau
        
        n = f_x.shape[0]
        mask = torch.eye(2 * n, device=f_x.device, dtype=torch.bool)
        logits.masked_fill_(mask, -float('inf'))
        
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        
        labels = torch.cat([torch.arange(n, 2 * n), torch.arange(n)], dim=0).to(f_x.device)
        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=2 * n).float()
        
        residuals = probabilities - one_hot_labels
     
        return residuals.flatten()

    def _apply_preconditioner(self):
        """
        Applies the Gauss-Newton preconditioner using CG with an efficient HVP.
        """
        if self.param_A.grad is None:
            return

        current_A = self.param_A.data
        grad_A = self.param_A.grad
        

        def h_gn_mvp(d_vec):
            d_matrix = d_vec.reshape(current_A.shape)
            _, jvp_result = jvp(self._compute_residuals, (current_A,), (d_matrix,))
            vjp_result_tuple = vjp(self._compute_residuals, current_A)[3](jvp_result)
            vjp_result = vjp_result_tuple
            
            hvp_flat = vjp_result.flatten()
            damped_hvp = hvp_flat + self.damping * d_vec
            return damped_hvp

        preconditioned_grad_flat = conjugate_gradient_solver(
            h_gn_mvp, grad_A.flatten(), max_iter=self.cg_max_iter
        )
        
        self.param_A.grad.copy_(preconditioned_grad_flat.reshape(grad_A.shape))

class SimclrGNHPreconditioner(BasePreconditioner):

    def __init__(self, model, params_to_precondition, temperature, damping=1e-3, cg_max_iter=10):
        super().__init__(model, params_to_precondition)
        if len(self.params)!= 1:
            raise ValueError("designed for a single matrix parameter (model.A).")
        
        self.param_A = self.params[0]
        self.tau = temperature
        self.damping = damping
        self.cg_max_iter = cg_max_iter
        self.K_nm = None
        self.K_nm_pos = None

    def _update_preconditioner(self, K_A=None, K_B=None, **kwargs):

        if K_A is None or K_B is None:
            raise ValueError("K_A (K_nm) and K_B (K_nm_pos) must be provided.")
        
        self.K_nm = K_A
        self.K_nm_pos = K_B

    def _compute_residuals(self, A_matrix):

        f_x = torch.matmul(self.K_nm, A_matrix)
        f_x_pos = torch.matmul(self.K_nm_pos, A_matrix)
        
        f_x = torch.nn.functional.normalize(f_x, p=2, dim=1)
        f_x_pos = torch.nn.functional.normalize(f_x_pos, p=2, dim=1)
        
        all_embeddings = torch.cat([f_x, f_x_pos], dim=0)
        logits = (all_embeddings @ all_embeddings.T) / self.tau
        
        n = f_x.shape[0]
        mask = torch.eye(2 * n, device=f_x.device, dtype=torch.bool)
        logits.masked_fill_(mask, -float('inf'))

        probabilities = torch.nn.functional.softmax(logits, dim=1)
        labels = torch.cat([torch.arange(n, 2 * n), torch.arange(n)], dim=0).to(f_x.device)
        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=2 * n).float()
        
        residuals = probabilities - one_hot_labels
     
        return  residuals.flatten()

    def _apply_preconditioner(self):
        if self.param_A.grad is None:
            return

        current_A = self.param_A.data
        grad_A = self.param_A.grad
        
        def h_gn_mvp(d_vec):
            d_matrix = d_vec.reshape(current_A.shape)
            _, jvp_result = jvp(self._compute_residuals, (current_A,), (d_matrix,))
            vjp_result_tuple = vjp(self._compute_residuals, current_A)[1](jvp_result)
            vjp_result = vjp_result_tuple
            
            hvp_flat = vjp_result[0].flatten()
            damped_hvp = hvp_flat + self.damping * d_vec
            return damped_hvp

        preconditioned_grad_flat = conjugate_gradient_solver(
            h_gn_mvp, grad_A.flatten(), max_iter=self.cg_max_iter
        )
        
        self.param_A.grad.copy_(preconditioned_grad_flat.reshape(grad_A.shape))
