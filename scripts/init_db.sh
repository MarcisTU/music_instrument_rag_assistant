#!/bin/bash
set -e

# These variables are automatically available if defined in your
# docker-compose 'environment' or 'env_file' sections.
# We use defaults (the :- syntax) to prevent the script from crashing.

POSTGRES_USER="${POSTGRES_USER}"
DB_NAME="${DB_NAME}"
DB_USER="${DB_USER}"
DB_PASS="${DB_PASSWORD}"

echo "Starting database initialization..."

# Perform initialization using the internal 'postgres' superuser
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "postgres" <<-EOSQL
    -- 1. Create the application user if it doesn't exist
    DO \$$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '$DB_USER') THEN
            CREATE ROLE $DB_USER WITH LOGIN PASSWORD '$DB_PASS';
        END IF;
    END
    \$$;

    -- 2. Create the database and assign ownership
    SELECT 'CREATE DATABASE $DB_NAME'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$DB_NAME')\gexec

    -- 3. Grant privileges
    GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;

    -- 4. Enable pgvector extension on the target database
    \c $DB_NAME
    CREATE EXTENSION IF NOT EXISTS vector;

    -- Ensure the user can create schemas/tables
    GRANT ALL ON SCHEMA public TO $DB_USER;
EOSQL

echo "Database '$DB_NAME' and user '$DB_USER' are ready with pgvector enabled."