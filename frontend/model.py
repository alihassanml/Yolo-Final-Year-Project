from sqlalchemy import Boolean,Integer,Column,String,ForeignKey,Float,DateTime, Enum, Text,LargeBinary
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from database import Base

class prediction(Base):
    __tablename__ = 'prediction'
    id = Column(Integer,primary_key=True,index=True)
    name = Column(String,unique=True,index=True)
    accuracy = Column(String,unique=True,index=True)
    time = Column(String,index=True)


    
    