<RSC Version="0.1">
    <Simulator>
        <Integration>
            <DtFrac>1</DtFrac>
        </Integration>
        <Damping>
            <InternalDampingZ>0.5</InternalDampingZ>
            <CollisionDampingZ>0.2</CollisionDampingZ>
            <GlobalDampingZ>0.01</GlobalDampingZ>
        </Damping>
        <RigidSolver>
            <RigidIterations>10</RigidIterations>
            <BaumgarteRatio>0.01</BaumgarteRatio>
        </RigidSolver>
        <Condition>
            <ResultStartCondition>
                t >= 0
            </ResultStartCondition>
            <ResultStopCondition>
                t >= 3
            </ResultStopCondition>
            <StopCondition>
                t >= 3
            </StopCondition>
        </Condition>
        <Gravity>
            <GravAcc>-9.8</GravAcc>
            <FloorEnabled>1</FloorEnabled>
        </Gravity>
        <!--        <FloorElevation>-->
        <!--            <X_Size>0.3</X_Size>-->
        <!--            <Y_Size>0.3</Y_Size>-->
        <!--            <X_Values>2</X_Values>-->
        <!--            <Y_Values>2</Y_Values>-->
        <!--            <Height>-->
        <!--                0, 0.05, 0, 0.05-->
        <!--            </Height>-->
        <!--        </FloorElevation>-->
    </Simulator>
    <Voxel>
        <Size>0.01</Size>
        <Palette>
            <Material ID="1">
                <Name>Body</Name>
                <Display>
                    <Red>1</Red>
                    <Green>0</Green>
                    <Blue>0</Blue>
                    <Alpha>0.2</Alpha>
                </Display>
                <Mechanical>
                    <MatModel>0</MatModel><!--0 = no failing-->
                    <ElasticMod>1e5</ElasticMod>
                    <Density>800</Density>
                    <PoissonsRatio>0.35</PoissonsRatio>
                    <FrictionStatic>1</FrictionStatic>
                    <FrictionDynamic>0.5</FrictionDynamic>
                    <MaxExpansion>0.5</MaxExpansion>
                    <MinExpansion>-0.5</MinExpansion>
                </Mechanical>
            </Material>
            <Material ID="2">
                <Name>Body</Name>
                <Display>
                    <Red>0</Red>
                    <Green>0</Green>
                    <Blue>1</Blue>
                    <Alpha>0.75</Alpha>
                </Display>
                <Mechanical>
                    <MatModel>0</MatModel><!--0 = no failing-->
                    <ElasticMod>1e5</ElasticMod>
                    <Density>1500</Density>
                    <PoissonsRatio>0.35</PoissonsRatio>
                    <FrictionStatic>1</FrictionStatic>
                    <FrictionDynamic>0.5</FrictionDynamic>
                    <MaxExpansion>0.5</MaxExpansion>
                    <MinExpansion>-0.5</MinExpansion>
                </Mechanical>
            </Material>
        </Palette>
    </Voxel>
</RSC>
