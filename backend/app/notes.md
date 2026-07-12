# 各文件具体职责

- database.py 中的 Base  
  **负责收集设计图**

- models.py 中的 Workspace  
  **是一张具体设计图**  

- filenest.db 中的 workspaces  
  **是真正建成的表**

- Session 是 SQLAlchemy ORM 中“一次数据库工作”的管理者。
  - Session 是一个工作台：
  - 在这个工作台上查询、添加和修改 ORM 对象，最后决定提交还是撤销这些操作。
  ```
  SessionFactory  
        ↓ 调用  
  SessionFactory()   
        ↓ 产生   
  一个新的 Session  
  ```

Swagger 只是开发阶段的接口测试台，  
在前端还没出现时，就能先确认后端业务是否正确。

