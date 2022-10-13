from fastapi import APIRouter

router = APIRouter(prefix='/dummy_hello', tags=['dummy'])

@router.get('/')
async def hello():
    return 'Hello Dummy!'