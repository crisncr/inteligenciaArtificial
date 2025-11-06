"""add_source_and_external_api_id_to_analyses

Revision ID: 23146ae4fc3a
Revises: 
Create Date: 2025-11-06 14:55:53.534706

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '23146ae4fc3a'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Agregar columna source con valor por defecto 'manual'
    op.add_column('analyses', sa.Column('source', sa.String(), nullable=False, server_default='manual'))
    
    # Agregar columna external_api_id como nullable
    op.add_column('analyses', sa.Column('external_api_id', sa.Integer(), nullable=True))
    
    # Agregar foreign key constraint
    op.create_foreign_key(
        'fk_analyses_external_api_id',
        'analyses',
        'external_apis',
        ['external_api_id'],
        ['id'],
        ondelete='SET NULL'
    )


def downgrade() -> None:
    # Eliminar foreign key constraint
    op.drop_constraint('fk_analyses_external_api_id', 'analyses', type_='foreignkey')
    
    # Eliminar columnas
    op.drop_column('analyses', 'external_api_id')
    op.drop_column('analyses', 'source')

