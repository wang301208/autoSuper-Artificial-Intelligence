"""堆排序算法实现。"""
from ...base import Algorithm
from typing import List, Any


class HeapSort(Algorithm):
    """使用堆排序算法对列表进行排序。

    堆排序通过构建最大堆，将最大元素逐个移到数组末尾，从而实现排序。
    设置 ``reverse=True`` 可返回降序结果。
    """

    def __init__(self, reverse: bool = False) -> None:
        self.reverse = reverse

    def execute(self, data: List[Any]) -> List[Any]:
        """返回数据的排序副本。

        参数:
            data: 待排序的数据列表

        返回:
            List[Any]: 排序后的数据副本

        时间复杂度: O(n log n)
        空间复杂度: O(1)
        """
        arr = data.copy()  # 创建数据副本，避免修改原数组
        n = len(arr)
        # 建堆
        for i in range(n // 2 - 1, -1, -1):
            self._heapify(arr, n, i)
        # 逐个将堆顶元素移到末尾
        for i in range(n - 1, 0, -1):
            arr[i], arr[0] = arr[0], arr[i]
            self._heapify(arr, i, 0)
        if self.reverse:
            arr.reverse()
        return arr

    def _heapify(self, arr: List[Any], n: int, i: int) -> None:
        """维护以 ``i`` 为根的最大堆性质。"""
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n and arr[left] > arr[largest]:
            largest = left
        if right < n and arr[right] > arr[largest]:
            largest = right

        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            self._heapify(arr, n, largest)
