"""示例源码：测试与演示用的固定任务输入。

三份小小的“列表函数”样本，覆盖三种真实写法：下标取值、属性取值、`.get()` 取值。
它们只用于离线测试和演示，让 extract 有真实的候选可选，不做任何文件读写。
"""

from __future__ import annotations

# 字典风格 + docstring：条件取自同一个字典键（active / deleted）。
USERS_MODULE = '''"""用户列表示例：字典风格。"""

USERS = [
    {"id": 1, "name": "Ada", "active": True, "deleted": False},
    {"id": 2, "name": "Bob", "active": False, "deleted": False},
]


def active_users(users):
    """返回有效用户，保持原顺序。"""
    result = []
    for user in users:
        if user["active"] and not user["deleted"]:
            result.append({"id": user["id"], "name": user["name"]})
    return result
'''

# 对象属性风格：条件与字段都是属性读取。
ORDERS_MODULE = '''"""订单列表示例：对象属性风格。"""

RATE = 0.1


class Order:
    """一条订单记录。"""

    def __init__(self, order_id, total, paid, shipped):
        self.order_id = order_id
        self.total = total
        self.paid = paid
        self.shipped = shipped


def paid_orders(orders):
    """返回已付款订单的编号与金额。"""
    rows = []
    for order in orders:
        if order.paid and not order.shipped:
            rows.append({"order_id": order.order_id, "total": order.total})
    return rows
'''

# 混合取值风格：同一个函数里既有 .get()、也有下标。
PRODUCTS_MODULE = '''"""商品列表示例：混合取值风格。"""


def visible_products(products):
    """返回上架商品的编号与价格。"""
    out = []
    for product in products:
        if product.get("visible"):
            out.append({"sku": product["sku"], "price": product.get("price")})
    return out
'''

# 没有任何函数：extract 应当拒绝。
NO_FUNCTION_MODULE = '''"""只有常量，没有函数。"""

TITLE = "没有函数"
COUNT = 3
'''

# 语法不合法：括号没有闭合，extract 应当拒绝。
BROKEN_MODULE = '''"""括号没有闭合。"""


def broken(items):
    result = []
    for item in items:
        if item["ok":
            result.append({"id": item["id"]})
    return result
'''
