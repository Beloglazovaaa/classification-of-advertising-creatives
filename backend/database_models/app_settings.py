from database import Base
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String


class AppSettings(Base):
    __tablename__ = "app_settings"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True, nullable=False)
    value = Column(String, nullable=False)
    description = Column(String, nullable=True)

    def get_value(self):
        if self.value.lower() in ("true", "false"):
            return self.value.lower() == "true"
        try:
            return int(self.value)
        except ValueError:
            try:
                return float(self.value)
            except ValueError:
                return self.value

    def set_value(self, new_value):
        self.value = str(new_value)
