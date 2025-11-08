"""add_routes_and_route_points_tables

Revision ID: 37fc866ca7fe
Revises: 23146ae4fc3a
Create Date: 2025-11-08 15:54:58.627932

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '37fc866ca7fe'
down_revision = '23146ae4fc3a'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Crear tabla routes
    op.create_table(
        'routes',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('algorithm', sa.String(), nullable=False, server_default='astar'),
        sa.Column('distance', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_routes_id'), 'routes', ['id'], unique=False)
    op.create_foreign_key(
        'fk_routes_user_id',
        'routes',
        'users',
        ['user_id'],
        ['id'],
        ondelete='CASCADE'
    )
    
    # Crear tabla route_points
    op.create_table(
        'route_points',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('route_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('address', sa.String(), nullable=False),
        sa.Column('lat', sa.Float(), nullable=False),
        sa.Column('lng', sa.Float(), nullable=False),
        sa.Column('display_name', sa.String(), nullable=True),
        sa.Column('order', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_route_points_id'), 'route_points', ['id'], unique=False)
    op.create_foreign_key(
        'fk_route_points_route_id',
        'route_points',
        'routes',
        ['route_id'],
        ['id'],
        ondelete='CASCADE'
    )


def downgrade() -> None:
    # Eliminar tabla route_points
    op.drop_constraint('fk_route_points_route_id', 'route_points', type_='foreignkey')
    op.drop_index(op.f('ix_route_points_id'), table_name='route_points')
    op.drop_table('route_points')
    
    # Eliminar tabla routes
    op.drop_constraint('fk_routes_user_id', 'routes', type_='foreignkey')
    op.drop_index(op.f('ix_routes_id'), table_name='routes')
    op.drop_table('routes')

