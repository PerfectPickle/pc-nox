from .base import NodeBase

class PcnConvNode(NodeBase):
    """Static convolutional node (sublayer) with independent spatial parameters 
    for parent (top-down, generative) and optional child (bottom-up, discriminative) 
    operations following bidirectional PC (Oliviers et al., 2026)

    td = top-down, generative, from parent
    bu = bottom-up, discriminative, from child
    """

    td_op: Union[eqx.nn.Linear, eqx.nn.ConvTranspose2d]
    bu_op: Union[eqx.nn.Conv2d, eqx.nn.Linear]

    own_shape: Tuple[int, int, int] = eqx.field(static=True)
    is_td_dense: bool = eqx.field(static=True)
    is_bu_dense: bool = eqx.field(static=True)
    self.supports_bottom_up

    td_act_fn: Callable = eqx.field(static=True)
    bu_act_fn: Callable = eqx.field(static=True)

    def __init__(
        self,
        parent_shape: Tuple[int, ...],
        own_shape: Tuple[int, int, int],
        child_shape: Tuple[int, ...],
        # --- Parent (Top-Down / ConvTranspose) Config ---
        td_kernel_size: int = 3,
        td_stride: int = 2,
        td_padding: Union[str, int] = 1,
        td_output_padding: Union[str, int] = 1,
        # --- Child (Bottom-Up / Conv2d) Config ---
        bu_kernel_size: int = 3,
        bu_stride: int = 2,
        bu_padding: Union[str, int] = 1,
        td_act_fn: Callable = jnp.tanh,
        bu_act_fn: Callable = jnp.tanh,
        bidirectional: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        k1, k2 = jr.split(key)
        self.own_shape = own_shape
        self.td_act_fn = td_act_fn
        self.bu_act_fn = bu_act_fn

        # -------------------------------------------------------------
        # 1. PARENT PROJECTION (Top-Down Generative)
        # -------------------------------------------------------------
        self.is_td_dense = (len(parent_shape) == 1)
        if self.is_td_dense:
            self.td_op = eqx.nn.Linear(
                in_features=parent_shape[0],
                out_features=prod(own_shape),
                use_bias=True,
                key=k1,
            )
        else:
            self.td_op = eqx.nn.ConvTranspose2d(
                in_channels=parent_shape[0],
                out_channels=own_shape[0],
                kernel_size=td_kernel_size,
                stride=td_stride,
                padding=td_padding,
                output_padding=td_output_padding,
                key=k1,
            )

        # -------------------------------------------------------------
        # 2. CHILD PROJECTION (Bottom-Up Discriminative Sweep)
        # -------------------------------------------------------------
        if bidirectional:
            self.is_bu_dense = (len(child_shape) == 1)
            if self.is_bu_dense:
                self.bu_op = eqx.nn.Linear(
                    in_features=child_shape[0],
                    out_features=int(jnp.prod(jnp.array(own_shape))),
                    use_bias=True,
                    key=k2,
                )
            else:
                self.bu_op = eqx.nn.Conv2d(
                    in_channels=bu_shape[0],
                    out_channels=own_shape[0],
                    kernel_size=bu_kernel_size,
                    stride=bu_stride,
                    padding=bu_padding,
                    key=k2,
                )
        else:
            self.bu_op = None


    # top-down, generative
    def predict_td(self, parent_curr: Array) -> Array:
        pre_act = self.parent_op(parent_curr)
        if self.is_td_dense:
            pre_act = jnp.reshape(pre_act, self.own_shape)
        return self.td_act_fn(pre_act)

    # bottom-up, discriminative
    def predict_bu(self, child_curr: Array) -> Array:
        if self.bu_op is None:
            raise RuntimeError("Bottom-up prediction is disabled.")
        pre_act = self.bu_op(child_curr)
        if self.is_bu_dense:
            pre_act = jnp.reshape(pre_act, self.own_shape)
        return self.bu_act_fn(pre_act)

    # little wrapper for testing
    def predict(self, parent_curr: Array) -> Array:
        self.predict_td(parent_curr)

    def weights(self) -> Tuple[Array, ...]:
        if self.bu_op is None:
            return (self.td_op.weight,)
        return (self.td_op.weight, self.bu_op.weight)
