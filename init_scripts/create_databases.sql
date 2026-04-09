SELECT 'CREATE DATABASE creatives_db' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'creatives_db')\gexec
SELECT 'CREATE DATABASE test_db' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'test_db')\gexec

GRANT ALL PRIVILEGES ON DATABASE creatives_db TO "user";
GRANT ALL PRIVILEGES ON DATABASE test_db TO "user";
