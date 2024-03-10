# -*- encoding: utf-8 -*-
# @Author: chaodreaming
# @Contact: chaodreaming@gmail.com
import traceback
from pathlib import Path
from typing import List
from typing import Tuple, Union

import numpy as np
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

root_dir = Path(__file__).resolve().parent
InputType = Union[str, np.ndarray, bytes, Path]
class OrtInferSession:
    def __init__(self, model_path: Union[str, Path], num_threads: int = 1):
        self.verify_exist(model_path)

        self.num_threads = num_threads
        self._init_sess_opt()

        EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        try:
            self.session = InferenceSession(
                str(model_path), sess_options=self.sess_opt, providers=EP_list
            )
        except TypeError:
            # 这里兼容ort 1.5.2
            self.session = InferenceSession(str(model_path), sess_options=self.sess_opt)

    def _init_sess_opt(self):
        self.sess_opt = SessionOptions()
        self.sess_opt.log_severity_level = 4
        self.sess_opt.enable_cpu_mem_arena = False

        if self.num_threads != -1:
            self.sess_opt.intra_op_num_threads = self.num_threads

        self.sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    def __call__(self, input_content: List[np.ndarray]) -> np.ndarray:
        input_dict = dict(zip(self.get_input_names(), input_content))
        # print("input_dict",input_dict)
        try:
            return self.session.run(None, input_dict)
        except Exception as e:
            error_info = traceback.format_exc()
            raise ONNXRuntimeError(error_info) from e

    def get_input_names(
        self,
    ):
        return [v.name for v in self.session.get_inputs()]

    def get_output_name(self, output_idx=0):
        return self.session.get_outputs()[output_idx].name

    def get_metadata(self):
        meta_dict = self.session.get_modelmeta().custom_metadata_map
        return meta_dict

    @staticmethod
    def verify_exist(model_path: Union[Path, str]):
        if not isinstance(model_path, Path):
            model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exist!")

        if not model_path.is_file():
            raise FileExistsError(f"{model_path} must be a file")

class Decoder:
    def __init__(self, decoder_path: Union[Path, str],max_seq_len=512):
        self.max_seq_len = max_seq_len
        self.session = OrtInferSession(decoder_path)

    def __call__(
        self,
        start_tokens,
        seq_len=256,
        eos_token=None,
        temperature=0.00001,
        filter_thres=0.9,
        context=None,
    ):
        num_dims = len(start_tokens.shape)

        b, t = start_tokens.shape

        out = start_tokens

        confidences = []
        for _ in range(seq_len):
            x = out[:, -self.max_seq_len :]


            ort_outs = self.session([x.astype(np.int64), context])[0]
            np_preds = ort_outs
            np_logits = np_preds[:, -1, :]
            sample = np.argmax(np_logits, axis=-1)
            # 获取最大值的索引


            # 使用np.expand_dims沿相同的轴保持维度
            np_probs = self.softmax(np_logits, axis=-1)
            confidence = np.max(np_probs, axis=1)
            confidences.append(confidence)  # 使用append而不是np.concatenate
            sample = np.expand_dims(sample, axis=-1)
            out = np.concatenate([out, sample], axis=-1)


            if (
                eos_token is not None
                and (np.cumsum(out == eos_token, axis=1)[:, -1] >= 1).all()
            ):
                break

        out = out[:, t:]
        if num_dims == 1:
            out = out.squeeze(0)
        return (out,[round(np.mean(i)*100,2) for i in confidences])

    @staticmethod
    def softmax(x, axis=None) -> float:
        def logsumexp(a, axis=None, b=None, keepdims=False):
            a_max = np.amax(a, axis=axis, keepdims=True)

            if a_max.ndim > 0:
                a_max[~np.isfinite(a_max)] = 0
            elif not np.isfinite(a_max):
                a_max = 0

            tmp = np.exp(a - a_max)

            # suppress warnings about log of zero
            with np.errstate(divide="ignore"):
                s = np.sum(tmp, axis=axis, keepdims=keepdims)
                out = np.log(s)

            if not keepdims:
                a_max = np.squeeze(a_max, axis=axis)
            out += a_max
            return out

        return np.exp(x - logsumexp(x, axis=axis, keepdims=True))

    def npp_top_k(self, logits, thres=0.9):
        k = int((1 - thres) * logits.shape[-1])
        val, ind = self.np_top_k(logits, k)
        probs = np.full_like(logits, float("-inf"))
        np.put_along_axis(probs, ind, val, axis=1)
        return probs

    @staticmethod
    def np_top_k(
        a: np.ndarray, k: int, axis=-1, largest=True, sorted=True
    ) -> Tuple[np.ndarray, np.ndarray]:
        if axis is None:
            axis_size = a.size
        else:
            axis_size = a.shape[axis]

        assert 1 <= k <= axis_size

        a = np.asanyarray(a)
        if largest:
            index_array = np.argpartition(a, axis_size - k, axis=axis)
            topk_indices = np.take(index_array, -np.arange(k) - 1, axis=axis)
        else:
            index_array = np.argpartition(a, k - 1, axis=axis)
            topk_indices = np.take(index_array, np.arange(k), axis=axis)

        topk_values = np.take_along_axis(a, topk_indices, axis=axis)
        if sorted:
            sorted_indices_in_topk = np.argsort(topk_values, axis=axis)
            if largest:
                sorted_indices_in_topk = np.flip(sorted_indices_in_topk, axis=axis)
            sorted_topk_values = np.take_along_axis(
                topk_values, sorted_indices_in_topk, axis=axis
            )
            sorted_topk_indices = np.take_along_axis(
                topk_indices, sorted_indices_in_topk, axis=axis
            )
            return sorted_topk_values, sorted_topk_indices
        return topk_values, topk_indices

    @staticmethod
    def multinomial(weights, num_samples, replacement=True):
        weights = np.asarray(weights)
        weights /= np.sum(weights)  # 确保权重之和为1
        indices = np.arange(len(weights))
        samples = np.random.choice(
            indices, size=num_samples, replace=replacement, p=weights
        )
        return samples
class ONNXRuntimeError(Exception):
    pass