from transformers import PretrainedConfig


class DeepseekV2Config(PretrainedConfig):
    model_type = "deepseek_v2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=102400,
        hidden_size=5120,
        intermediate_size=12288,
        moe_intermediate_size=1408,
        num_hidden_layers=60,
        num_attention_heads=128,
        max_position_embeddings=163840,
        rms_norm_eps=1e-6,
        rope_theta=10000,
        attention_bias=False,
        q_lora_rank=None,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        n_shared_experts=2,
        n_routed_experts=64,
        num_experts_per_tok=6,
        first_k_dense_replace=1,
        scoring_func="softmax",
        norm_topk_prob=True,
        architectures=None,
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.scoring_func = scoring_func
        self.norm_topk_prob = norm_topk_prob
        if architectures is None:
            architectures = ["DeepseekV2ForCausalLM"]
        super().__init__(
            architectures=architectures,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
