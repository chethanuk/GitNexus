public class Caller {
    private void testCallGraph3() {}
    private void testCallGraph2() {
        testCallGraph3();
        System.out.println(TestConstants.TEST);
        System.out.println(TestConstants.TEST_2);
        TestConstants testConstants = new TestConstants();
        System.out.println(testConstants.TEST_3);
        System.out.println("Today is " + ViewCommEnum.DEP_TYPCD_101.getName());
        System.out.println(ViewCommEnum.DEP_TYPCD_103);
    }
}
